"""
Comprehensive evaluation for Bhojpuri Transformer: LM metrics, generation, attention analysis.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.transformer import BhojpuriTransformer
from train.dataset import PackedLMDataset
from tokenizer.tokenizer_wrapper import BhojpuriTokenizer


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    model = BhojpuriTransformer()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def evaluate_lm_metrics(model, val_loader, device):
    """Compute cross-entropy loss, PPL, BPB on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_bytes = 0

    tokenizer = BhojpuriTokenizer()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, return_attn=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")

            total_loss += loss.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return {"loss": avg_loss, "ppl": ppl}


def generate_samples(model, tokenizer, device, num_samples=3, max_new_tokens=50):
    """Generate text samples at different temperatures."""
    model.eval()

    samples = {}
    for temp in [0.5, 1.0, 1.5]:
        temps_samples = []
        for i in range(num_samples):
            prompt_ids = torch.tensor([[2] + [i % 100 for i in range(10)]], device=device)

            with torch.no_grad():
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    greedy=False
                )

            text = tokenizer.decode(generated[0].cpu().tolist())
            temps_samples.append(text)

        samples[f"temp_{temp}"] = temps_samples

    return samples


def analyze_attention(model, tokenizer, device, layer_idx=0, num_heads=2):
    """Extract and visualize attention patterns."""
    model.eval()

    prompt_text = "भोजपुरी भाषा"
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=device)

    with torch.no_grad():
        logits, attn_list = model(prompt_ids, return_attn=True)

    if attn_list is None or layer_idx >= len(attn_list):
        return None

    attn = attn_list[layer_idx]
    batch_size, seq_len, seq_len_k = attn.shape[0] // model.num_heads, attn.shape[1], attn.shape[2]

    attention_entropy = []
    for h in range(min(num_heads, model.num_heads)):
        head_attn = attn[h]
        entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-10), dim=1).mean().item()
        attention_entropy.append(entropy)

    return {"layer": layer_idx, "entropy": attention_entropy, "seq_len": seq_len}


def main(checkpoint_path, val_data_path):
    """Main evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model...")
    model = load_model(checkpoint_path, device)
    print(f"✓ Loaded model with {model.count_parameters():,} parameters")

    print("\nLoading tokenizer...")
    tokenizer = BhojpuriTokenizer()

    print("\nLoading validation data...")
    val_dataset = PackedLMDataset(val_data_path, max_seq_len=model.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("\n" + "="*60)
    print("LANGUAGE MODELING EVALUATION")
    print("="*60)

    lm_metrics = evaluate_lm_metrics(model, val_loader, device)
    print(f"\nValidation Loss: {lm_metrics['loss']:.4f}")
    print(f"Perplexity: {lm_metrics['ppl']:.2f}")

    print("\n" + "="*60)
    print("GENERATION SAMPLES")
    print("="*60)

    samples = generate_samples(model, tokenizer, device, num_samples=3)
    for temp_key, temp_samples in samples.items():
        print(f"\n{temp_key}:")
        for i, s in enumerate(temp_samples, 1):
            print(f"  {i}. {s[:100]}...")

    print("\n" + "="*60)
    print("ATTENTION ANALYSIS")
    print("="*60)

    for layer in [0, model.num_layers - 1]:
        attn_info = analyze_attention(model, tokenizer, device, layer_idx=layer, num_heads=4)
        if attn_info:
            print(f"\nLayer {attn_info['layer']}:")
            print(f"  Attention entropy (first 4 heads): {attn_info['entropy']}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--val-data", default="../data/val.bin")
    args = parser.parse_args()

    main(args.checkpoint, args.val_data)
