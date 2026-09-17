#!/usr/bin/env python3
"""
Real finetuning loss/PPL curves for the three finetuned models (Telugu high-parameter, Telugu
low-parameter, Bhojpuri high-parameter -- Bhojpuri low-parameter was never finetuned, see
report/final_report.tex Sec 4.1). Reads the real training logs directly (JSONL:
step/epoch/train_loss/val_loss/val_ppl/lr/timestamp), same format/loader as
report/phase-2/code/plot_training_curves.py -- no synthetic/placeholder data.

Each finetuning run shows the overfitting pattern already documented in the report's tables:
train_loss keeps falling toward ~0 across all 20 epochs while val_loss/val_ppl bottoms out early
(epoch 1-3) then rises -- the reason checkpoint_best.pt (lowest val_loss), not the final epoch,
is used for the reported test-set accuracy. The best epoch is marked on each curve.

Usage: python3 report/phase-3/code/plot_finetune_curves.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "report" / "phase-3" / "plots" / "finetune_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "telugu_high": dict(
        log=ROOT / "telugu/model/outputs/finetune_checkpoints/training_telugu.log",
        label="Telugu high-parameter finetune (best epoch 3, 2.50% test acc.)",
        color="#ff7f0e",
    ),
    "telugu_low": dict(
        log=ROOT / "telugu/model/outputs/finetune_checkpoints_low/training_telugu.log",
        label="Telugu low-parameter finetune (best epoch 2, 8.75% test acc.)",
        color="#1f77b4",
    ),
    "bhojpuri_high": dict(
        log=ROOT / "bhojpuri/model/outputs/finetune_checkpoints/training_bhojpuri.log",
        label="Bhojpuri high-parameter finetune (best epoch 1, 16.40% test acc.)",
        color="#d62728",
    ),
}


def load_log(path):
    decoder = json.JSONDecoder()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            pos = 0
            while pos < len(line):
                obj, end = decoder.raw_decode(line, pos)
                rows.append(obj)
                pos = end
                while pos < len(line) and line[pos].isspace():
                    pos += 1
    return rows


def best_epoch(rows):
    best = min(rows, key=lambda r: r["val_loss"])
    return best["epoch"]


def plot_all_models_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key, cfg in MODELS.items():
        rows = data[key]
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r["train_loss"] for r in rows], marker="o", markersize=2,
                 linestyle="-", label=f"{cfg['label']} -- train", color=cfg["color"])
        ax.plot(epochs, [r["val_loss"] for r in rows], marker="s", markersize=2,
                 linestyle="--", color=cfg["color"], alpha=0.6)
        be = best_epoch(rows)
        best_row = next(r for r in rows if r["epoch"] == be)
        ax.scatter([be], [best_row["val_loss"]], marker="*", s=140, color=cfg["color"],
                    edgecolor="black", zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Finetuning: Train (solid) vs. Val (dashed) Loss, All 3 Models\n(★ = best epoch, checkpoint_best.pt)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, cfg in MODELS.items():
        rows = data[key]
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r["val_ppl"] for r in rows], marker="o", markersize=3,
                 label=cfg["label"], color=cfg["color"])
        be = best_epoch(rows)
        best_row = next(r for r in rows if r["epoch"] == be)
        ax.scatter([be], [best_row["val_ppl"]], marker="*", s=140, color=cfg["color"],
                    edgecolor="black", zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity (PPL)")
    ax.set_yscale("log")
    ax.set_title("Finetuning: Validation PPL, All 3 Models (log scale)\n(★ = best epoch, checkpoint_best.pt)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "all_finetune_models_comparison.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.relative_to(ROOT)}")


def main():
    data = {}
    for key, cfg in MODELS.items():
        rows = load_log(cfg["log"])
        data[key] = rows
        be = best_epoch(rows)
        best_row = next(r for r in rows if r["epoch"] == be)
        print(f"Loaded {len(rows)} epochs from {cfg['log'].relative_to(ROOT)} "
              f"(best epoch {be}: val_loss={best_row['val_loss']}, val_ppl={best_row['val_ppl']})")

    plot_all_models_comparison(data)


if __name__ == "__main__":
    main()
