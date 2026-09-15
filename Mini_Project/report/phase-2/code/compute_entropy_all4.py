#!/usr/bin/env python3
"""
Real output-distribution entropy (bits, log2) at the next-token position after temperature
scaling, for all four pretrained models, replacing report/phase-2/report.md Sec 6.3's
fabricated entropy table (that section's numbers were never computed from any real model --
same fabrication pattern as the rest of the original report). Computed from real forward
passes over 30 real held-out lines per model at each of 4 temperatures.

Usage: python3 report/phase-2/code/compute_entropy_all4.py
(Run from the repo root with venv/bin/python3.)
"""

import sys, json, importlib
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer

ROOT = "."
LOW_PARAM_ARCH = dict(vocab_size=10000, d_model=256, num_layers=6, num_heads=8, d_ff=1024, max_seq_len=128)
TELUGU_HIGH_PARAM_ARCH = dict(vocab_size=20000, d_model=384, num_layers=10, num_heads=8, d_ff=1536, max_seq_len=256)
BHOJPURI_HIGH_PARAM_ARCH = dict(vocab_size=16000, d_model=320, num_layers=8, num_heads=8, d_ff=1280, max_seq_len=256)

VARIANTS = [
    dict(key="telugu_low", label="Telugu low-param", module="telugu.model.transformer", cls="TeluguTransformer", arch=LOW_PARAM_ARCH,
         ckpt="telugu/model/outputs/submission/checkpoint_best.pt",
         tok="SidLMA/telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json", test="telugu/data/test/telugu.txt"),
    dict(key="telugu_high", label="Telugu high-param", module="telugu.model.transformer", cls="TeluguTransformer", arch=TELUGU_HIGH_PARAM_ARCH,
         ckpt="telugu/model/outputs/checkpoints/checkpoint_best.pt",
         tok="telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json", test="telugu/data/test/telugu.txt"),
    dict(key="bhojpuri_low", label="Bhojpuri low-param", module="bhojpuri.model.transformer", cls="BhojpuriTransformer", arch=LOW_PARAM_ARCH,
         ckpt="bhojpuri/model/outputs/checkpoints_sub/checkpoint_best.pt",
         tok="SidLMA/bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json", test="bhojpuri/data/test/bhoj.txt"),
    dict(key="bhojpuri_high", label="Bhojpuri high-param", module="bhojpuri.model.transformer", cls="BhojpuriTransformer", arch=BHOJPURI_HIGH_PARAM_ARCH,
         ckpt="bhojpuri/model/outputs/checkpoints/checkpoint_best.pt",
         tok="bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json", test="bhojpuri/data/test/bhoj.txt"),
]

def load_lines(path, n=30, min_len=40, max_len=160):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if min_len <= len(line) <= max_len:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines

results = {}
for v in VARIANTS:
    mod = importlib.import_module(v["module"])
    cls = getattr(mod, v["cls"])
    model = cls(**v["arch"])
    ckpt = torch.load(v["ckpt"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = HFTokenizer.from_file(v["tok"])
    lines = load_lines(v["test"])

    per_temp = {}
    for T in [0.5, 1.0, 1.5, 2.0]:
        entropies = []
        with torch.no_grad():
            for text in lines:
                ids = tok.encode(text).ids[:15]
                if len(ids) < 3:
                    continue
                input_ids = torch.tensor([ids])
                logits, _ = model(input_ids)
                last_logits = logits[0, -1] / T
                probs = F.softmax(last_logits, dim=-1)
                ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log2()).sum().item()
                entropies.append(ent)
        mean_ent = sum(entropies) / len(entropies)
        std_ent = (sum((e - mean_ent) ** 2 for e in entropies) / len(entropies)) ** 0.5
        per_temp[str(T)] = {"mean": round(mean_ent, 4), "std": round(std_ent, 4)}
        print(f"{v['key']} T={T}: entropy_mean={mean_ent:.4f} bits, std={std_ent:.4f}")
    results[v["key"]] = {"label": v["label"], "vocab_size": v["arch"]["vocab_size"], "entropy_by_temp": per_temp}

with open("report/phase-2/metrics/entropy_all4.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved report/phase-2/metrics/entropy_all4.json")
