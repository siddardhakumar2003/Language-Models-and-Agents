#!/usr/bin/env python3
"""
Real Phase 2 attention analysis -- replaces the fabricated heatmaps that used to live at
report/phase-2/plots/attention_complete/. The old generate_plots.py (repo root) built those
with np.random.randn plus a hand-coded "local attention bias" multiplier and never ran the
model; report/phase-2/report.md Sec 6.4's numbers were hand-matched to that fake data. This
script produces the real thing: every heatmap and every entropy/distance number here comes
from an actual forward pass (model(input_ids, return_attn=True)) on real trained checkpoints.

Telugu ablation (all heads, both a "low_parameter_model" and "high_parameter_model" variant --
report/phase-3/report.md Sec 3 has the full writeup):
  - low_parameter_model:  telugu/model/outputs/submission/checkpoint_best.pt
                          (6 layers, 10K vocab, 7.34M params, PPL 881.9 -- architecture-matched
                          to Bhojpuri's submission checkpoint for a clean H-vs-L comparison)
  - high_parameter_model: telugu/model/outputs/checkpoints/checkpoint_best.pt
                          (10 layers, 20K vocab, 25.5M params, PPL ~4120 as of writing, still
                          training -- the larger Telugu-only architecture the project continued
                          training past the submission checkpoint)
Bhojpuri only has one variant here (its submission/checkpoints_sub checkpoint is currently the
project's best Bhojpuri model, so there's no "low vs high" split to run for it).

Tokenizer note: the low_parameter_model variants need the ORIGINAL 10K-vocab tokenizer, which
the current telugu/tokenizer/ and bhojpuri/tokenizer/ files are NOT (they were later retrained
to larger 20K/16K vocabularies -- incompatible with the submission checkpoints' embedding
tables). The original 10K tokenizer files are preserved in the SidLMA/ mirror directory and
used here for the low_parameter_model variants; high_parameter_model uses the current 20K
tokenizer, which is what that checkpoint was actually trained with.

Output:
  report/phase-2/plots/attention_complete/telugu/low_parameter_model/*.png
  report/phase-2/plots/attention_complete/telugu/high_parameter_model/*.png
  report/phase-2/plots/attention_complete/bhojpuri/*.png                      (unchanged layout)
  report/phase-2/plots/attention_complete/complete_attention_metrics.json     (telugu nested by variant)

Usage: python3 report/phase-2/code/real_attention_analysis.py
"""

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
from tokenizers import Tokenizer as HFTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "report" / "phase-2" / "plots" / "attention_complete"

for _font_path in [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
]:
    if Path(_font_path).exists():
        fm.fontManager.addfont(_font_path)

# Verified from each checkpoint's own checkpoint["config"]["model_config"] (Telugu) / the
# current configs/model_config.json (Bhojpuri high-parameter, matches its checkpoint's actual
# state_dict shapes).
LOW_PARAM_ARCH = dict(vocab_size=10000, d_model=256, num_layers=6, num_heads=8, d_ff=1024, max_seq_len=128)
TELUGU_HIGH_PARAM_ARCH = dict(vocab_size=20000, d_model=384, num_layers=10, num_heads=8, d_ff=1536, max_seq_len=256)
BHOJPURI_HIGH_PARAM_ARCH = dict(vocab_size=16000, d_model=320, num_layers=8, num_heads=8, d_ff=1280, max_seq_len=256)

VARIANTS = [
    dict(
        lang="telugu", variant_dir="low_parameter_model",
        model_module="telugu.model.transformer", model_cls="TeluguTransformer",
        model_arch=LOW_PARAM_ARCH,
        ckpt=ROOT / "telugu/model/outputs/submission/checkpoint_best.pt",
        tokenizer_path=ROOT / "SidLMA/telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json",
        test_txt=ROOT / "telugu/data/test/telugu.txt",
        label="Telugu (Model H) -- low-parameter (7.34M, 6L, 10K vocab, submission checkpoint)",
        file_key="telugu_(h)_low", metrics_key=("telugu_h", "low_parameter_model"), font="Noto Sans Telugu",
    ),
    dict(
        lang="telugu", variant_dir="high_parameter_model",
        model_module="telugu.model.transformer", model_cls="TeluguTransformer",
        model_arch=TELUGU_HIGH_PARAM_ARCH,
        ckpt=ROOT / "telugu/model/outputs/checkpoints/checkpoint_best.pt",
        tokenizer_path=ROOT / "telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json",
        test_txt=ROOT / "telugu/data/test/telugu.txt",
        label="Telugu (Model H) -- high-parameter (25.5M, 10L, 20K vocab, in-progress checkpoint)",
        file_key="telugu_(h)_high", metrics_key=("telugu_h", "high_parameter_model"), font="Noto Sans Telugu",
    ),
    dict(
        # Bhojpuri's high-parameter checkpoint -- historically the ONLY Bhojpuri variant this
        # script analyzed (its submission/low-param PPL, 870.1, and this one's, 814.5, are close
        # enough that there was originally no meaningful low-vs-high *ablation* story worth
        # running). Output path/file_key/metrics_key are UNCHANGED from before (variant_dir=None,
        # flat bhojpuri/ layout) so existing references in report/phase-2/report.md and
        # report/final_report.tex keep working; a dedicated low_parameter_model variant was added
        # below (nested, new files only) once all-4-models heatmaps were needed for the
        # consolidated final report.
        lang="bhojpuri", variant_dir=None,
        model_module="bhojpuri.model.transformer", model_cls="BhojpuriTransformer",
        model_arch=BHOJPURI_HIGH_PARAM_ARCH,
        ckpt=ROOT / "bhojpuri/model/outputs/checkpoints/checkpoint_best.pt",
        tokenizer_path=ROOT / "bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json",
        test_txt=ROOT / "bhojpuri/data/test/bhoj.txt",  # real held-out corpus text, not synthetic
        label="Bhojpuri (Model L)", file_key="bhojpuri_(l)", metrics_key=("bhojpuri_l",), font="Noto Sans Devanagari",
    ),
    dict(
        # Architecture-matched to Telugu's low_parameter_model (same LOW_PARAM_ARCH): the
        # original submission checkpoint, 6 layers, 10K vocab, 7.34M params. Needs the ORIGINAL
        # 10K-vocab tokenizer (SidLMA/ mirror), same reasoning as Telugu's low-parameter variant
        # above -- the current bhojpuri/tokenizer/ was later retrained to 16K vocab.
        lang="bhojpuri", variant_dir="low_parameter_model",
        model_module="bhojpuri.model.transformer", model_cls="BhojpuriTransformer",
        model_arch=LOW_PARAM_ARCH,
        ckpt=ROOT / "bhojpuri/model/outputs/submission_phase-2/checkpoint_best.pt",
        tokenizer_path=ROOT / "SidLMA/bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json",
        test_txt=ROOT / "bhojpuri/data/test/bhoj.txt",
        label="Bhojpuri (Model L) -- low-parameter (7.34M, 6L, 10K vocab, submission checkpoint)",
        file_key="bhojpuri_(l)_low", metrics_key=("bhojpuri_l", "low_parameter_model"), font="Noto Sans Devanagari",
    ),
]

NUM_ENTROPY_SAMPLES = 40


def load_model(module_name, cls_name, model_arch, ckpt_path, device):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    model = cls(**model_arch)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt.get("step"), ckpt.get("best_val_ppl")


def attn_entropy_per_head(attn):
    p = attn.clamp_min(1e-12)
    return (-(p * p.log()).sum(dim=-1)).mean(dim=-1)


def attn_mean_distance_per_head(attn):
    num_heads, seq, _ = attn.shape
    pos = torch.arange(seq, dtype=torch.float32)
    dist = (pos.view(-1, 1) - pos.view(1, -1)).abs()
    return (attn * dist.unsqueeze(0)).sum(dim=-1).mean(dim=-1)


def plot_all_heads(attn, meta_line, prompt_line, out_path, font=None):
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
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_avg(attn, meta_line, prompt_line, out_path, font=None):
    avg = attn.mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(avg, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    fig.suptitle(meta_line, fontsize=9, fontweight="bold", y=0.99)
    fig.text(0.5, 0.90, prompt_line, ha="center", fontsize=8, fontfamily=font, wrap=True)
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def load_test_lines(path, n=200, min_len=40, max_len=160):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if min_len <= len(line) <= max_len:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines


def analyze_variant(cfg, device):
    print(f"\n{'='*70}\n{cfg['label']} -- REAL Phase 2 attention analysis\n{'='*70}")

    tokenizer = HFTokenizer.from_file(str(cfg["tokenizer_path"]))
    print(f"✓ Loaded tokenizer (vocab={tokenizer.get_vocab_size()}) from {cfg['tokenizer_path'].relative_to(ROOT)}")

    model, ckpt_step, ckpt_ppl = load_model(cfg["model_module"], cfg["model_cls"], cfg["model_arch"], cfg["ckpt"], device)
    ppl_str = f"{ckpt_ppl:.2f}" if ckpt_ppl is not None else "n/a"
    print(f"✓ Loaded checkpoint {cfg['ckpt'].relative_to(ROOT)} (step {ckpt_step}, best_val_ppl {ppl_str}, {model.count_parameters():,} params)")

    lines = load_test_lines(cfg["test_txt"])
    print(f"✓ Loaded {len(lines)} real test sentences from {cfg['test_txt'].relative_to(ROOT)}")

    out_dir = OUT_DIR / cfg["lang"] / cfg["variant_dir"] if cfg["variant_dir"] else OUT_DIR / cfg["lang"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Heatmaps: one real sentence, ALL layers, all_heads + avg ----
    example_text = lines[0]
    ids = tokenizer.encode(example_text).ids
    input_ids = torch.tensor([ids], device=device)
    print(f"  Heatmap example ({len(ids)} tokens): {example_text!r}")

    with torch.no_grad():
        _, attn_list = model(input_ids, return_attn=True)

    for layer_idx in range(model.num_layers):
        attn = attn_list[layer_idx].cpu()
        meta_line = f"{cfg['label']} -- Layer {layer_idx} Attention Weights ({attn.shape[0]} heads)"
        prompt_line = example_text[:100]
        plot_all_heads(attn, meta_line, prompt_line,
                        out_dir / f"{cfg['file_key']}_layer{layer_idx}_all_heads.png", font=cfg["font"])
        plot_avg(attn, meta_line + " (head-averaged)", prompt_line,
                  out_dir / f"{cfg['file_key']}_layer{layer_idx}_avg.png", font=cfg["font"])
    print(f"  ✓ Saved {model.num_layers} layers x (all_heads + avg) to {out_dir.relative_to(ROOT)}")

    # ---- Entropy / mean distance, averaged over NUM_ENTROPY_SAMPLES real sentences ----
    sample_lines = lines[:NUM_ENTROPY_SAMPLES]
    per_layer_entropy = [[] for _ in range(model.num_layers)]
    per_layer_distance = [[] for _ in range(model.num_layers)]
    for text in sample_lines:
        ids = tokenizer.encode(text).ids
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            _, attn_list = model(input_ids, return_attn=True)
        for layer_idx in range(model.num_layers):
            attn = attn_list[layer_idx].cpu()
            per_layer_entropy[layer_idx].append(attn_entropy_per_head(attn))
            per_layer_distance[layer_idx].append(attn_mean_distance_per_head(attn))

    layer_stats = {}
    for layer_idx in range(model.num_layers):
        ent = torch.stack(per_layer_entropy[layer_idx])
        dist = torch.stack(per_layer_distance[layer_idx])
        layer_stats[str(layer_idx)] = {
            "entropy_mean": round(ent.mean().item(), 4),
            "entropy_std": round(ent.std().item(), 4),
            "distance_mean": round(dist.mean().item(), 4),
            "distance_std": round(dist.std().item(), 4),
            "num_heads": ent.shape[1],
        }
    layer_stats["_meta"] = {
        "checkpoint": str(cfg["ckpt"].relative_to(ROOT)),
        "checkpoint_step": ckpt_step,
        "checkpoint_best_val_ppl": ckpt_ppl,
        "params": model.count_parameters(),
        "num_layers": model.num_layers,
    }
    print(f"  ✓ Computed entropy/distance over {len(sample_lines)} real sentences x {model.num_layers} layers")
    return cfg["metrics_key"], layer_stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["telugu", "bhojpuri", "both"], default="both",
                         help="Which language(s) to (re)analyze. Existing complete_attention_metrics.json entries for the other language are preserved.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_path = OUT_DIR / "complete_attention_metrics.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    variants = VARIANTS if args.lang == "both" else [v for v in VARIANTS if v["lang"] == args.lang]
    for cfg in variants:
        metrics_key, layer_stats = analyze_variant(cfg, device)
        node = results
        for part in metrics_key[:-1]:
            node = node.setdefault(part, {})
        node[metrics_key[-1]] = layer_stats

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
