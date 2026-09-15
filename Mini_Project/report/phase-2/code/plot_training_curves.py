#!/usr/bin/env python3
"""
Loss curves + consolidated raw log values for all 4 pretrained model variants:
  - Telugu low-parameter  (telugu/model/outputs/submission/training_telugu.log)
  - Telugu high-parameter (telugu/model/outputs/checkpoints/training_telugu.log)
  - Bhojpuri low-parameter  (bhojpuri/model/outputs/checkpoints_sub/training_bhojpuri.log)
  - Bhojpuri high-parameter (bhojpuri/model/outputs/checkpoints/training_bhojpuri.log)

Reads the real training logs directly (JSONL: step/epoch/train_loss/val_loss/val_ppl/lr/
timestamp) -- no synthetic/placeholder data. For each model: a per-model loss-curve figure
(train/val loss + val PPL, titled/labeled/legended per the project spec). Also two low-vs-high
overlay comparisons (one per language) and one 4-way comparison, plus a consolidated
JSON/CSV dump of every raw log line from all four logs for the report appendix.

Usage: python3 report/phase-2/code/plot_training_curves.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "report" / "phase-2" / "plots" / "training_curves"
METRICS_DIR = ROOT / "report" / "phase-2" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "telugu_low": dict(
        log=ROOT / "telugu/model/outputs/submission/training_telugu.log",
        label="Telugu low-parameter (6L, 10K vocab, 7.34M params)",
        color="#1f77b4",
    ),
    "telugu_high": dict(
        log=ROOT / "telugu/model/outputs/checkpoints/training_telugu.log",
        label="Telugu high-parameter (10L, 20K vocab, 25.5M params)",
        color="#ff7f0e",
    ),
    "bhojpuri_low": dict(
        log=ROOT / "bhojpuri/model/outputs/checkpoints_sub/training_bhojpuri.log",
        label="Bhojpuri low-parameter (6L, 10K vocab, 7.34M params)",
        color="#2ca02c",
    ),
    "bhojpuri_high": dict(
        log=ROOT / "bhojpuri/model/outputs/checkpoints/training_bhojpuri.log",
        label="Bhojpuri high-parameter (8L, 16K vocab, 15.1M params)",
        color="#d62728",
    ),
}


def load_log(path):
    """Parses JSONL, but tolerates lines with more than one JSON object concatenated with no
    separator (seen in telugu/model/outputs/checkpoints/training_telugu.log line 18 -- a
    session-resume artifact where a new session appended without a leading newline)."""
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


def plot_single_model(key, cfg, rows):
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    val_ppl = [r["val_ppl"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", markersize=3, label="Train loss", color="#4c72b0")
    ax.plot(epochs, val_loss, marker="o", markersize=3, label="Val loss", color="#dd8452")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{cfg['label']}\nTraining/Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, val_ppl, marker="o", markersize=3, color="#55a868", label="Val perplexity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity (PPL)")
    ax.set_title(f"{cfg['label']}\nValidation Perplexity")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / f"{key}_loss_curve.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.relative_to(ROOT)}  ({len(rows)} epochs, final val_ppl={val_ppl[-1]:.2f})")


def plot_language_comparison(lang, low_key, high_key, data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key in (low_key, high_key):
        rows = data[key]
        ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows],
                marker="o", markersize=3, label=MODELS[key]["label"], color=MODELS[key]["color"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"{lang.capitalize()}: Low- vs. High-Parameter Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key in (low_key, high_key):
        rows = data[key]
        ax.plot([r["epoch"] for r in rows], [r["val_ppl"] for r in rows],
                marker="o", markersize=3, label=MODELS[key]["label"], color=MODELS[key]["color"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity (PPL)")
    ax.set_yscale("log")
    ax.set_title(f"{lang.capitalize()}: Low- vs. High-Parameter Validation PPL (log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / f"{lang}_low_vs_high_comparison.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.relative_to(ROOT)}")


def plot_all_models_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key, cfg in MODELS.items():
        rows = data[key]
        ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows],
                marker="o", markersize=2, label=cfg["label"], color=cfg["color"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("All 4 Models: Validation Loss")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, cfg in MODELS.items():
        rows = data[key]
        ax.plot([r["epoch"] for r in rows], [r["val_ppl"] for r in rows],
                marker="o", markersize=2, label=cfg["label"], color=cfg["color"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity (PPL)")
    ax.set_yscale("log")
    ax.set_title("All 4 Models: Validation PPL (log scale)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "all_models_comparison.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.relative_to(ROOT)}")


def main():
    data = {}
    for key, cfg in MODELS.items():
        rows = load_log(cfg["log"])
        data[key] = rows
        print(f"Loaded {len(rows)} epochs from {cfg['log'].relative_to(ROOT)}")

    print()
    for key, cfg in MODELS.items():
        plot_single_model(key, cfg, data[key])

    print()
    plot_language_comparison("telugu", "telugu_low", "telugu_high", data)
    plot_language_comparison("bhojpuri", "bhojpuri_low", "bhojpuri_high", data)
    plot_all_models_comparison(data)

    # ---- Consolidated raw log values (every epoch, every model) ----
    consolidated = {key: data[key] for key in MODELS}
    json_path = METRICS_DIR / "training_logs_consolidated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n✓ {json_path.relative_to(ROOT)}")

    csv_path = METRICS_DIR / "training_logs_consolidated.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "epoch", "train_loss", "val_loss", "val_ppl", "lr", "timestamp"])
        for key, rows in data.items():
            for r in rows:
                writer.writerow([key, r["step"], r["epoch"], r["train_loss"], r["val_loss"], r["val_ppl"], r["lr"], r["timestamp"]])
    print(f"✓ {csv_path.relative_to(ROOT)}")

    # ---- Final-epoch summary table (quick reference) ----
    print("\nFinal-epoch summary:")
    print(f"{'model':<16} {'epochs':>7} {'final_step':>11} {'train_loss':>11} {'val_loss':>9} {'val_ppl':>10}")
    for key, rows in data.items():
        last = rows[-1]
        print(f"{key:<16} {len(rows):>7} {last['step']:>11} {last['train_loss']:>11} {last['val_loss']:>9} {last['val_ppl']:>10}")


if __name__ == "__main__":
    main()
