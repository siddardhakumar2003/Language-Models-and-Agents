"""
Bhojpuri pretraining entry point for the NO-POSITIONAL-EMBEDDINGS ablation
(LMA_Individual_Project_v1.pdf, Bonus section).

New file. Does NOT modify train/train.py -- that file's run_training() hardcodes an import of
the standard BhojpuriTransformer class at module level, so it can't be reused directly to train
a different model class. This file mirrors run_training()'s logic exactly, but imports
Trainer / load_config (from train.train) and PackedLMDataset (from train.dataset) and
BhojpuriTokenizer (from tokenizer.tokenizer_wrapper) unchanged -- zero duplication of the
actual training-loop/data-loading/checkpointing logic, only the model class and the config
file it reads architecture from differ from the standard run.

NOTE: sys.path is pointed at train/kaggle_bundle, NOT this file's own parent (bhojpuri/), on
purpose. `diff`-ing confirmed the top-level bhojpuri/train/train.py and bhojpuri/train/dataset.py
are a STALE, older, incompatible pair (step-based, .bin-file-based, Trainer.__init__ missing
ckpt_dir/base_model/total_steps, PackedLMDataset reading pre-tokenized .bin instead of raw .txt)
left over from an earlier version of the pipeline -- train/kaggle_bundle/train/{train,dataset}.py
is the current, correct, actually-used-on-Kaggle pair (epoch-based, raw .txt + on-the-fly
tokenization). This mirror (a top-level copy for visibility/consistency with the rest of the
repo's top-level/kaggle_bundle pairing convention) targets the bundle so it stays correct;
the canonical copy that actually gets uploaded to Kaggle is train/kaggle_bundle/train/train_no_pos.py.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_BUNDLE_ROOT = Path(__file__).parent / "kaggle_bundle"
sys.path.insert(0, str(_BUNDLE_ROOT))
from model.transformer_no_pos import BhojpuriTransformerNoPos
from train.dataset import PackedLMDataset
from train.train import Trainer, load_config
from tokenizer.tokenizer_wrapper import BhojpuriTokenizer


def run_training_no_pos(args, root_dir=None, data_dir=None, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root_dir = Path(root_dir) if root_dir is not None else Path(__file__).parent.parent
    data_dir = Path(data_dir) if data_dir is not None else root_dir / "data"
    config_dir = root_dir / "configs"

    config = load_config(str(config_dir))
    training_config = config["training_config"]

    override_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
    }
    for key, value in override_config.items():
        if value is not None:
            training_config[key] = value

    print("\nTraining config (no-positional-embeddings ablation):")
    for k, v in training_config.items():
        print(f"  {k}: {v}")

    tokenizer = BhojpuriTokenizer()
    print(f"✓ Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

    model_config_path = config_dir / "model_config_no_pos.json"
    model = BhojpuriTransformerNoPos(config_path=str(model_config_path), vocab_size=tokenizer.vocab_size)
    model = model.to(device)

    base_model = model
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"✓ Using {num_gpus} GPUs via DataParallel")
        model = nn.DataParallel(model)

    assert base_model.vocab_size == tokenizer.vocab_size, (
        f"model vocab_size {base_model.vocab_size} != tokenizer vocab_size {tokenizer.vocab_size}"
    )
    print(f"\n✓ Initialized BhojpuriTransformerNoPos with {base_model.count_parameters():,} parameters "
          f"(no positional embeddings -- {15082240 - base_model.count_parameters():,} fewer params than "
          f"the standard 15,082,240-param model)")

    print(f"\nPreparing data from: {data_dir}")
    train_txt = data_dir / "train" / "bhoj.txt"
    val_txt = data_dir / "val" / "bhoj.txt"

    if not train_txt.exists():
        print(f"✗ Training data not found at {train_txt}. Expected: data/train/bhoj.txt")
        sys.exit(1)

    train_dataset = PackedLMDataset(str(train_txt), tokenizer, max_seq_len=base_model.max_seq_len)
    val_dataset = PackedLMDataset(str(val_txt), tokenizer, max_seq_len=base_model.max_seq_len) if val_txt.exists() else train_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=training_config["batch_size"], shuffle=True,
        num_workers=training_config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    if output_dir is None:
        log_dir, ckpt_dir = "logs_no_pos", "checkpoints_no_pos"
    else:
        log_dir = ckpt_dir = str(Path(output_dir))

    num_epochs = training_config["num_epochs"]
    runtime_total_steps = len(train_loader) * num_epochs
    print(f"\nEpoch-based training: {len(train_loader)} steps/epoch x {num_epochs} epochs = {runtime_total_steps:,} total steps")

    trainer = Trainer(
        model, train_loader, val_loader, device, config,
        log_dir=log_dir, ckpt_dir=ckpt_dir, base_model=base_model, total_steps=runtime_total_steps,
    )
    # Distinct log filename so this ablation run never overwrites/confuses the standard run's log
    # if both happen to write into a shared directory at some point.
    trainer.log_file = Path(log_dir) / "training_bhojpuri_no_pos.log"

    if args.resume_from:
        if Path(args.resume_from).exists():
            trainer.load_checkpoint(args.resume_from)
        else:
            print(f"⚠️  resume_from={args.resume_from} was specified but not found -- starting from scratch (epoch 0).")
    else:
        print("No checkpoint specified -- starting from scratch (epoch 0).")

    start_epoch = trainer.current_epoch

    print(f"\n{'='*60}")
    print(f"Starting NO-POS training -- starting at epoch {start_epoch}, target {num_epochs} total epochs")
    print(f"{'='*60}")

    if start_epoch >= num_epochs:
        print(f"Checkpoint already at epoch {start_epoch} >= num_epochs {num_epochs}; nothing to train.")

    for epoch in range(start_epoch, num_epochs):
        avg_train_loss = trainer.train_epoch()
        val_loss, val_ppl = trainer.validate()
        trainer._log_metrics(
            step=trainer.current_step, epoch=trainer.current_epoch,
            train_loss=avg_train_loss, val_loss=val_loss, val_ppl=val_ppl,
            lr=trainer.optimizer.param_groups[0]["lr"],
        )
        print(f"Epoch {trainer.current_epoch}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.best_val_ppl = val_ppl
            trainer.best_step = trainer.current_step
            trainer.save_checkpoint(is_best=True)
        trainer.save_checkpoint(is_best=False)

    print(f"\n✓ Training complete!")
    print(f"  Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"  Best val_ppl: {trainer.best_val_ppl:.2f}")
    print(f"  Best checkpoint at step {trainer.best_step}")

    test_txt = data_dir / "test" / "bhoj.txt"
    if test_txt.exists():
        test_dataset = PackedLMDataset(str(test_txt), tokenizer, max_seq_len=base_model.max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=training_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
        test_loss, test_ppl = trainer.validate(test_loader)
        print(f"\n✓ Test set: loss={test_loss:.4f}, ppl={test_ppl:.2f}")
    else:
        print(f"\n⚠️  Test data not found at {test_txt}, skipping final test evaluation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--amp", type=bool, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default="checkpoints_no_pos/checkpoint_last.pt")
    args = parser.parse_args()

    if not Path(args.resume_from).exists():
        args.resume_from = None

    run_training_no_pos(args)
