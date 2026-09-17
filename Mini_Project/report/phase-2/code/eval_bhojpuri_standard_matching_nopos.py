#!/usr/bin/env python3
"""
Fills in the two blank cells in the no-positional-embeddings ablation table
(report/final_report.tex, Ablation Study 4): "Full test-set eval PPL / BPB" and "Greedy
repetition rate" for the STANDARD (with positional embeddings) Bhojpuri high-parameter model.

These were never computed for the standard model with the exact same protocol as the no-pos
ablation notebook's own eval cell (bhojpuri/train/pretrain_no_pos_kaggle.ipynb cell 7) -- this
script reproduces that same protocol (full real test set for PPL/BPB; PREFIX_CHARS=50,
REF_CHARS=100, greedy decoding for repetition rate) against the real standard checkpoint, so the
two rows are a genuine apples-to-apples comparison, not a proxy metric.

Usage: python3 report/phase-2/code/eval_bhojpuri_standard_matching_nopos.py
"""

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from bhojpuri.model.transformer import BhojpuriTransformer
from bhojpuri.tokenizer.tokenizer_wrapper import BhojpuriTokenizer

device = torch.device("cpu")
tokenizer = BhojpuriTokenizer()
model = BhojpuriTransformer().to(device)
ckpt = torch.load(ROOT / "bhojpuri/model/outputs/checkpoints/checkpoint_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded standard Bhojpuri checkpoint (step {ckpt.get('step')}, {model.count_parameters():,} params)")

with open(ROOT / "bhojpuri/data/test/bhoj.txt", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Loaded {len(lines):,} real test lines")

# ---- [1/2] Full test-set PPL / BPB (same protocol as the no-pos notebook cell 7) ----
t0 = time.time()
total_loss, total_tokens, n_samples = 0.0, 0, 0
with torch.no_grad():
    for i, text in enumerate(lines):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        ids = ids[:model.max_seq_len]
        input_ids = torch.tensor([ids[:-1]], device=device)
        target_ids = torch.tensor(ids[1:], device=device)
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits[0, :len(target_ids)], target_ids)
        total_loss += loss.item() * len(target_ids)
        total_tokens += len(target_ids)
        n_samples += 1
        if (i + 1) % 20000 == 0:
            print(f"  ...{i+1:,}/{len(lines):,} lines, {time.time()-t0:.0f}s elapsed")

avg_loss = total_loss / total_tokens
ppl = math.exp(min(avg_loss, 20))
bpb = avg_loss / math.log(2)
print(f"[1/2] PPL={ppl:.2f}  BPB={bpb:.4f}  samples={n_samples}  tokens={total_tokens:,}  ({time.time()-t0:.0f}s)")

# ---- [2/2] Greedy repetition rate (same PREFIX_CHARS/REF_CHARS protocol) ----
PREFIX_CHARS, REF_CHARS, NUM_GEN_SAMPLES = 50, 100, 1000
pairs = []
for text in lines:
    if len(text) < PREFIX_CHARS + 10:
        continue
    prefix, ref = text[:PREFIX_CHARS], text[PREFIX_CHARS:PREFIX_CHARS + REF_CHARS]
    if ref.strip():
        pairs.append((prefix, ref))
    if len(pairs) >= NUM_GEN_SAMPLES:
        break

def repetition_rate(texts, n=2):
    total, repeated = 0, 0
    for t in texts:
        toks = t.split()
        grams = list(zip(*[toks[i:] for i in range(n)]))
        if not grams:
            continue
        seen = Counter(grams)
        total += len(grams)
        repeated += sum(c - 1 for c in seen.values() if c > 1)
    return round(repeated / total, 4) if total else 0.0

hyps = []
with torch.no_grad():
    for prefix, ref in pairs:
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if not ids:
            continue
        ref_ids = tokenizer.encode(ref, add_special_tokens=False)
        n_new = max(5, min(len(ref_ids), 40))
        input_ids = torch.tensor([ids], device=device)
        generated = model.generate(input_ids, max_new_tokens=n_new, temperature=1.0, greedy=True)
        gen_ids = generated[0].cpu().numpy().tolist()[len(ids):]
        hyps.append(tokenizer.decode(gen_ids))

rep_rate = repetition_rate(hyps, n=2)
print(f"[2/2] Greedy repetition rate (bigram) = {rep_rate}  (n={len(hyps)} samples)")

result = {
    "checkpoint": "bhojpuri/model/outputs/checkpoints/checkpoint_best.pt",
    "full_test_set_ppl": round(ppl, 2),
    "full_test_set_bpb": round(bpb, 4),
    "samples_evaluated": n_samples,
    "total_tokens": total_tokens,
    "greedy_repetition_rate_bigram": rep_rate,
    "greedy_num_samples": len(hyps),
}
out_path = ROOT / "report/phase-2/metrics/bhojpuri_standard_matching_nopos_eval.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\n✓ Saved {out_path.relative_to(ROOT)}")
print(json.dumps(result, indent=2))
