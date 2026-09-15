#!/usr/bin/env python3
"""
Phase 3 Attention Analysis (Sec 3.2, PDF): pretrained vs. finetuned comparison.

Unlike report/phase-2/plots' generate_plots.py (which fabricated attention heatmaps with
np.random noise instead of running the model), every number and pixel here comes from a real
forward pass with return_attn=True on the actual trained checkpoints. No synthetic data.

For each language (telugu, bhojpuri):
  - Loads the pretrained checkpoint and the best finetuned checkpoint.
  - Picks a real comparative-reasoning prompt from the finetune test set.
  - Plots per-head heatmaps (early layer = 0, late layer = last) for both pretrained and
    finetuned, with proper titles/axis labels/colorbars and the actual prompt text shown.
  - Computes attention entropy and mean attention distance per head/layer, averaged over a
    sample of test prompts, for both pretrained and finetuned -- saved to attention_summary.json.
  - Classifies heads as local vs long-range (by mean attention distance vs. the sequence's
    midpoint) and reports how that split shifts between pretrained and finetuned.

Usage: python3 report/phase-3/code/attention_analysis.py [--language telugu|bhojpuri|both]
"""

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

# Register the Devanagari/Telugu-capable fonts so example sentences in plot titles render
# as real glyphs instead of missing-glyph boxes (the default DejaVu Sans has no coverage
# for these scripts).
for _font_path in [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
]:
    if Path(_font_path).exists():
        fm.fontManager.addfont(_font_path)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "report" / "phase-3"
PLOTS_DIR = OUT_DIR / "plots" / "attention"
METRICS_DIR = OUT_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LANG_CONFIG = {
    "telugu": dict(
        model_module="telugu.model.transformer", model_cls="TeluguTransformer",
        tok_module="telugu.tokenizer.tokenizer_wrapper", tok_cls="TeluguTokenizer",
        pretrained_ckpt=ROOT / "telugu/model/outputs/checkpoints/checkpoint_best.pt",
        finetuned_ckpt=ROOT / "telugu/finetune/outputs/finetune_checkpoints/checkpoint_best.pt",
        test_jsonl=ROOT / "telugu/finetune/data/test.jsonl",
        label="Telugu (Model H)",
        font="Noto Sans Telugu",
    ),
    "bhojpuri": dict(
        model_module="bhojpuri.model.transformer", model_cls="BhojpuriTransformer",
        tok_module="bhojpuri.tokenizer.tokenizer_wrapper", tok_cls="BhojpuriTokenizer",
        pretrained_ckpt=ROOT / "bhojpuri/model/outputs/checkpoints/checkpoint_best.pt",
        finetuned_ckpt=ROOT / "bhojpuri/finetune/outputs/finetune_checkpoints/checkpoint_best.pt",
        test_jsonl=ROOT / "bhojpuri/finetune/data/test.jsonl",
        label="Bhojpuri (Model L)",
        font="Noto Sans Devanagari",
    ),
}

NUM_ENTROPY_SAMPLES = 40  # prompts averaged over for the entropy/distance summary table


def load_model(module_name, cls_name, ckpt_path, device):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    model = cls()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    return model, step


def attn_entropy_per_head(attn):
    """attn: (num_heads, seq, seq) -> (num_heads,) entropy averaged over query positions."""
    p = attn.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)  # (num_heads, seq)
    return ent.mean(dim=-1)


def attn_mean_distance_per_head(attn):
    """attn: (num_heads, seq, seq) -> (num_heads,) |query_pos - key_pos| weighted by attention."""
    num_heads, seq, _ = attn.shape
    pos = torch.arange(seq, dtype=torch.float32)
    dist = (pos.view(-1, 1) - pos.view(1, -1)).abs()  # (seq_q, seq_k)
    weighted = attn * dist.unsqueeze(0)
    return weighted.sum(dim=-1).mean(dim=-1)


def plot_all_heads(attn, meta_line, prompt_line, out_path, font=None):
    """meta_line is plain ASCII (default font); prompt_line is the native-script example
    sentence, rendered separately in `font` since no single installed font covers both
    Latin and Telugu/Devanagari glyphs."""
    num_heads = attn.shape[0]
    cols = min(4, num_heads)
    rows = math.ceil(num_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h].numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}", fontsize=9)
        ax.set_xlabel("Key Position", fontsize=8)
        ax.set_ylabel("Query Position", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for h in range(num_heads, len(axes)):
        axes[h].axis("off")
    fig.suptitle(meta_line, fontsize=10, fontweight="bold", y=0.99)
    fig.text(0.5, 0.955, prompt_line, ha="center", fontsize=9, fontfamily=font, wrap=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_language(lang, device):
    cfg = LANG_CONFIG[lang]
    print(f"\n{'='*70}\n{cfg['label']} attention analysis\n{'='*70}")

    tok_mod = importlib.import_module(cfg["tok_module"])
    tokenizer = getattr(tok_mod, cfg["tok_cls"])()

    pretrained, pre_step = load_model(cfg["model_module"], cfg["model_cls"], cfg["pretrained_ckpt"], device)
    finetuned, ft_step = load_model(cfg["model_module"], cfg["model_cls"], cfg["finetuned_ckpt"], device)
    print(f"✓ Loaded pretrained (step {pre_step}) and finetuned (step {ft_step}) checkpoints")

    examples = [json.loads(l) for l in open(cfg["test_jsonl"], encoding="utf-8") if l.strip()]
    num_layers = pretrained.num_layers
    layers_to_plot = sorted(set([0, num_layers - 1]))

    # ---- Heatmaps on one real comparative-reasoning example (early + late layer) ----
    example = sorted(examples, key=lambda e: abs(len(e["prompt"]) - 90))[0]
    ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    print(f"  Heatmap example prompt ({len(ids)} tokens): {example['prompt']!r}")
    print(f"  Gold answer: {example['answer']!r}")

    stage_models = {"pretrained": pretrained, "finetuned": finetuned}
    for stage, model in stage_models.items():
        with torch.no_grad():
            _, attn_list = model(input_ids, return_attn=True)
        for layer_idx in layers_to_plot:
            attn = attn_list[layer_idx].cpu()  # (num_heads, seq, seq), batch=1
            which = "early" if layer_idx == layers_to_plot[0] else "late"
            meta_line = f"{cfg['label']} -- {stage} -- layer {layer_idx} ({which}), {attn.shape[0]} heads"
            prompt_line = example["prompt"].replace("\n", "  ")
            out_path = PLOTS_DIR / f"{lang}_{stage}_layer{layer_idx}_{which}_all_heads.png"
            plot_all_heads(attn, meta_line, prompt_line, out_path, font=cfg["font"])
            print(f"  ✓ {out_path.relative_to(ROOT)}")

    # ---- Entropy / mean-distance summary, averaged over NUM_ENTROPY_SAMPLES prompts ----
    sample_examples = examples[:NUM_ENTROPY_SAMPLES]
    summary = {"pretrained": {}, "finetuned": {}}
    for stage, model in stage_models.items():
        per_layer_entropy = [[] for _ in range(num_layers)]
        per_layer_distance = [[] for _ in range(num_layers)]
        for ex in sample_examples:
            ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                _, attn_list = model(input_ids, return_attn=True)
            for layer_idx in range(num_layers):
                attn = attn_list[layer_idx].cpu()
                per_layer_entropy[layer_idx].append(attn_entropy_per_head(attn))
                per_layer_distance[layer_idx].append(attn_mean_distance_per_head(attn))

        layer_stats = {}
        for layer_idx in range(num_layers):
            ent = torch.stack(per_layer_entropy[layer_idx])  # (n_examples, num_heads)
            dist = torch.stack(per_layer_distance[layer_idx])
            layer_stats[str(layer_idx)] = {
                "entropy_mean_per_head": [round(v, 4) for v in ent.mean(dim=0).tolist()],
                "entropy_std_per_head": [round(v, 4) for v in ent.std(dim=0).tolist()],
                "distance_mean_per_head": [round(v, 4) for v in dist.mean(dim=0).tolist()],
                "distance_std_per_head": [round(v, 4) for v in dist.std(dim=0).tolist()],
                "num_heads": ent.shape[1],
            }
        summary[stage] = layer_stats
        print(f"  ✓ Computed entropy/distance over {len(sample_examples)} prompts x {num_layers} layers ({stage})")

    # ---- Local vs long-range head classification (median mean-distance split) ----
    classification = {}
    for stage in ["pretrained", "finetuned"]:
        all_dist = []
        for layer_idx in range(num_layers):
            all_dist.extend(summary[stage][str(layer_idx)]["distance_mean_per_head"])
        median_dist = float(np.median(all_dist))
        n_local = sum(1 for d in all_dist if d < median_dist)
        n_long_range = len(all_dist) - n_local
        classification[stage] = {
            "median_distance": round(median_dist, 3),
            "n_heads_below_median": n_local,
            "n_heads_above_median": n_long_range,
            "overall_mean_distance": round(float(np.mean(all_dist)), 3),
        }

    out = {
        "language": lang,
        "pretrained_checkpoint_step": pre_step,
        "finetuned_checkpoint_step": ft_step,
        "num_layers": num_layers,
        "example_prompt": example["prompt"],
        "example_answer": example["answer"],
        "layer_summary": summary,
        "local_vs_long_range": classification,
    }
    out_path = METRICS_DIR / f"{lang}_attention_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {out_path.relative_to(ROOT)}")
    print(f"  Overall mean attention distance: pretrained={classification['pretrained']['overall_mean_distance']}, "
          f"finetuned={classification['finetuned']['overall_mean_distance']}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["telugu", "bhojpuri", "both"], default="both")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    langs = ["telugu", "bhojpuri"] if args.language == "both" else [args.language]
    for lang in langs:
        analyze_language(lang, device)


if __name__ == "__main__":
    main()
