"""
Telugu Transformer Pretraining Script
Trains a decoder-only Transformer LM from scratch with checkpoint resume support.
"""

import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.transformer import TeluguTransformer
from train.dataset import PackedLMDataset


def load_config(config_path):
    """Load model/training/tokenizer configs."""
    script_dir = Path(__file__).parent.parent

    configs = {}
    for fname in ["model_config.json", "training_config.json", "tokenizer_config.json"]:
        fpath = script_dir / "configs" / fname
        with open(fpath, "r") as f:
            configs[fname.replace(".json", "")] = json.load(f)

    return configs


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config, log_dir="logs"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_config = config["training_config"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )

        total_steps = self.training_config["total_steps"]
        warmup_steps = self.training_config["warmup_steps"]

        self.scheduler = self._get_scheduler(warmup_steps, total_steps)

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler() if self.training_config["amp"] and device.type == "cuda" else None

        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_ppl = float("inf")
        self.best_step = 0

        self.log_file = self.log_dir / "training_telugu.log"

    def _get_scheduler(self, warmup_steps, total_steps):
        """Cosine annealing with linear warmup."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, math.cos(math.pi * 0.5 * (current_step - warmup_steps) / (total_steps - warmup_steps)))

        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(input_ids, return_attn=False)
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config["max_grad_norm"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(input_ids, return_attn=False)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config["max_grad_norm"])
                self.optimizer.step()

            self.scheduler.step()
            self.current_step += 1

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            if self.current_step % self.training_config["eval_steps"] == 0:
                val_loss, val_ppl = self.validate()
                self._log_metrics(
                    step=self.current_step,
                    epoch=self.current_epoch,
                    train_loss=total_loss / (len(pbar)),
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    lr=self.optimizer.param_groups[0]["lr"],
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_ppl = val_ppl
                    self.best_step = self.current_step
                    self.save_checkpoint(is_best=True)

            if self.current_step % self.training_config["checkpoint_steps"] == 0:
                self.save_checkpoint(is_best=False)

            if self.current_step >= self.training_config["total_steps"]:
                break

        self.current_epoch += 1
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.model(input_ids, return_attn=False)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    def _log_metrics(self, step, epoch, train_loss, val_loss, val_ppl, lr):
        """Log metrics to JSONL file."""
        log_entry = {
            "step": step,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "lr": f"{lr:.2e}",
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def save_checkpoint(self, is_best=False):
        """Save checkpoint with full resume capability."""
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.current_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": self.best_val_ppl,
            "best_step": self.best_step,
            "config": self.config,
        }

        if is_best:
            ckpt_path = ckpt_dir / "checkpoint_best.pt"
            print(f"✓ Saving best checkpoint to {ckpt_path} (step {self.current_step}, val_loss={self.best_val_loss:.4f})")
        else:
            ckpt_path = ckpt_dir / "checkpoint_last.pt"
            print(f"  Saving checkpoint to {ckpt_path}")

        torch.save(ckpt_dict, ckpt_path)

    def load_checkpoint(self, ckpt_path):
        """Resume from checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_step = ckpt["step"]
        self.current_epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_val_ppl = ckpt["best_val_ppl"]
        self.best_step = ckpt["best_step"]
        print(f"✓ Resumed from checkpoint: step {self.current_step}, epoch {self.current_epoch}")


def main(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(None)
    training_config = config["training_config"]

    override_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "eval_steps": args.eval_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "amp": args.amp,
    }
    for key, value in override_config.items():
        if value is not None:
            training_config[key] = value

    print(f"\nTraining config:")
    for k, v in training_config.items():
        print(f"  {k}: {v}")

    model = TeluguTransformer()
    model = model.to(device)
    print(f"\n✓ Initialized TeluguTransformer with {model.count_parameters():,} parameters")

    print(f"\nPreparing data...")
    data_dir = Path(__file__).parent.parent / "data"

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    if not train_bin.exists():
        print(f"✗ Training data not found at {train_bin}. Run prepare_bin.py first.")
        sys.exit(1)

    train_dataset = PackedLMDataset(str(train_bin), max_seq_len=model.max_seq_len)
    val_dataset = PackedLMDataset(str(val_bin), max_seq_len=model.max_seq_len) if val_bin.exists() else train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    trainer = Trainer(model, train_loader, val_loader, device, config, log_dir="logs")

    if args.resume_from and Path(args.resume_from).exists():
        trainer.load_checkpoint(args.resume_from)

    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")

    while trainer.current_step < training_config["total_steps"]:
        avg_loss = trainer.train_epoch()
        print(f"Epoch {trainer.current_epoch}: avg_loss={avg_loss:.4f}")

    print(f"\n✓ Training complete!")
    print(f"  Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"  Best val_ppl: {trainer.best_val_ppl:.2f}")
    print(f"  Best checkpoint at step {trainer.best_step}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, default=None)
    parser.add_argument("--amp", type=bool, default=None)
    parser.add_argument("--resume-from", type=str, default="checkpoints/checkpoint_last.pt",
                        help="path to checkpoint (default: checkpoints/checkpoint_last.pt)")
    args = parser.parse_args()

    if Path(args.resume_from).exists():
        args.resume_from = args.resume_from
    else:
        args.resume_from = None

    main(args)
