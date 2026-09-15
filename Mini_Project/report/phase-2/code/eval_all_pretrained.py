#!/usr/bin/env python3
"""
Real PPL/BPB + generation-quality (BLEU/chrF/ROUGE-L) + diversity/repetition evaluation for
all 4 pretrained model variants (Telugu low/high-parameter, Bhojpuri low/high-parameter).

This is deliberately a from-scratch script, not a reuse of the repo's existing
generate_model_samples.py / report/phase-2 report numbers: generate_model_samples.py's
"samples" are hand-typed fake data (its own docstring says "Create realistic sample data
matching the models trained" -- the token ids don't even decode to the "generated_text" shown
next to them), and report.md's existing PPL/BLEU/chrF numbers (541.96 PPL etc.) don't match any
real checkpoint's actual perplexity verified elsewhere in this project, so they're not trusted
either. Every number here comes from an actual forward pass / generate() call on the real
checkpoints and real held-out corpus text -- no synthetic data.

For each of the 4 variants:
  - PPL / BPB on real held-out test lines (BPB normalized by real UTF-8 byte length, not token
    count, per the project spec's "bits-per-byte ... to make the two languages more comparable
    when tokenizers differ").
  - Generation from real held-out prefixes at temperatures 0.5 / 1.0 / 1.5, scored against the
    real continuation of that same held-out line (BLEU-4, chrF, ROUGE-L via sacrebleu / rouge_score).
  - Diversity (Distinct-1/2) and repetition rate (fraction of repeated bigrams) on the
    generated text at each temperature.
  - A handful of real generated samples saved verbatim for the report.

Usage: python3 report/phase-2/code/eval_all_pretrained.py
(Run with venv/bin/python3 -- the system python3 doesn't have sacrebleu/rouge_score installed.)
"""

import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import sacrebleu
import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

METRICS_DIR = ROOT / "report" / "phase-2" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

from tokenizers import Tokenizer as HFTokenizer

LOW_PARAM_ARCH = dict(vocab_size=10000, d_model=256, num_layers=6, num_heads=8, d_ff=1024, max_seq_len=128)
TELUGU_HIGH_PARAM_ARCH = dict(vocab_size=20000, d_model=384, num_layers=10, num_heads=8, d_ff=1536, max_seq_len=256)
BHOJPURI_HIGH_PARAM_ARCH = dict(vocab_size=16000, d_model=320, num_layers=8, num_heads=8, d_ff=1280, max_seq_len=256)

VARIANTS = [
    dict(
        key="telugu_low", lang="telugu",
        model_module="telugu.model.transformer", model_cls="TeluguTransformer", model_arch=LOW_PARAM_ARCH,
        ckpt=ROOT / "telugu/model/outputs/submission/checkpoint_best.pt",
        tokenizer_path=ROOT / "SidLMA/telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json",
        test_txt=ROOT / "telugu/data/test/telugu.txt",
        label="Telugu low-parameter (6L, 10K vocab, PPL 881.9 checkpoint)",
    ),
    dict(
        key="telugu_high", lang="telugu",
        model_module="telugu.model.transformer", model_cls="TeluguTransformer", model_arch=TELUGU_HIGH_PARAM_ARCH,
        ckpt=ROOT / "telugu/model/outputs/checkpoints/checkpoint_best.pt",
        tokenizer_path=ROOT / "telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json",
        test_txt=ROOT / "telugu/data/test/telugu.txt",
        label="Telugu high-parameter (10L, 20K vocab, in-progress checkpoint)",
    ),
    dict(
        key="bhojpuri_low", lang="bhojpuri",
        model_module="bhojpuri.model.transformer", model_cls="BhojpuriTransformer", model_arch=LOW_PARAM_ARCH,
        ckpt=ROOT / "bhojpuri/model/outputs/checkpoints_sub/checkpoint_best.pt",
        tokenizer_path=ROOT / "SidLMA/bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json",
        test_txt=ROOT / "bhojpuri/data/test/bhoj.txt",
        label="Bhojpuri low-parameter (6L, 10K vocab, PPL 870.1 checkpoint)",
    ),
    dict(
        key="bhojpuri_high", lang="bhojpuri",
        model_module="bhojpuri.model.transformer", model_cls="BhojpuriTransformer", model_arch=BHOJPURI_HIGH_PARAM_ARCH,
        ckpt=ROOT / "bhojpuri/model/outputs/checkpoints/checkpoint_best.pt",
        tokenizer_path=ROOT / "bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json",
        test_txt=ROOT / "bhojpuri/data/test/bhoj.txt",
        label="Bhojpuri high-parameter (8L, 16K vocab, PPL 814.5 checkpoint)",
    ),
]

TEMPERATURES = [0.5, 1.0, 1.5]
NUM_PPL_LINES = 300
NUM_GEN_PREFIXES = 30
PROMPT_TOKENS = 15
CONTINUATION_TOKENS = 25
SEED = 42


def load_model(cfg, device):
    mod = importlib.import_module(cfg["model_module"])
    cls = getattr(mod, cfg["model_cls"])
    model = cls(**cfg["model_arch"])
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_test_lines(path, n, min_len=60, max_len=200):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if min_len <= len(line) <= max_len:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines


def compute_ppl_bpb(model, tokenizer, lines, device):
    total_nll_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    with torch.no_grad():
        for text in lines:
            ids = tokenizer.encode(text).ids
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids[:-1]], device=device)
            targets = torch.tensor(ids[1:], device=device)
            logits, _ = model(input_ids)
            logits = logits[0]
            n = min(logits.shape[0], targets.shape[0])
            loss = torch.nn.functional.cross_entropy(logits[:n], targets[:n], reduction="sum")
            total_nll_nats += loss.item()
            total_tokens += n
            total_bytes += len(text.encode("utf-8"))
    avg_nll_per_token = total_nll_nats / total_tokens
    ppl = math.exp(min(avg_nll_per_token, 20))
    bpb = (total_nll_nats / math.log(2)) / total_bytes
    return {"ppl": round(ppl, 4), "bpb": round(bpb, 4), "avg_nll_nats": round(avg_nll_per_token, 4),
            "tokens_evaluated": total_tokens, "lines_evaluated": len(lines)}


def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def repetition_rate(tokens, n=2):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams) if ngrams else 0.0


def evaluate_variant(cfg, device):
    print(f"\n{'='*70}\n{cfg['label']}\n{'='*70}")
    tokenizer = HFTokenizer.from_file(str(cfg["tokenizer_path"]))
    model = load_model(cfg, device)
    print(f"✓ Loaded model ({model.count_parameters():,} params) + tokenizer (vocab={tokenizer.get_vocab_size()})")

    ppl_lines = load_test_lines(cfg["test_txt"], NUM_PPL_LINES)
    ppl_bpb = compute_ppl_bpb(model, tokenizer, ppl_lines, device)
    print(f"✓ PPL={ppl_bpb['ppl']:.2f}  BPB={ppl_bpb['bpb']:.4f}  ({ppl_bpb['tokens_evaluated']} tokens over {ppl_bpb['lines_evaluated']} real lines)")

    gen_lines = load_test_lines(cfg["test_txt"], NUM_PPL_LINES + NUM_GEN_PREFIXES)[NUM_PPL_LINES:NUM_PPL_LINES + NUM_GEN_PREFIXES]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    temp_results = {}
    sample_generations = {}
    for temp in TEMPERATURES:
        hyps, refs = [], []
        examples = []
        torch.manual_seed(SEED)
        for text in gen_lines:
            encoding = tokenizer.encode(text)
            ids, tokens = encoding.ids, encoding.tokens
            if len(ids) < PROMPT_TOKENS + 5:
                continue
            # Advance the prompt/reference split point forward to the next real word-start (a
            # token NOT prefixed with WordPiece's "##" continuation marker), so the prompt
            # never ends mid-word and the reference never begins mid-word. Without this, a
            # split that lands inside a word makes ref_text start with a stray "##" once
            # decoded in isolation -- an artifact of where we cut, unrelated to what the model
            # actually generates, since a fresh generation only rarely starts with "##" itself.
            split = PROMPT_TOKENS
            while split < len(tokens) and tokens[split].startswith("##"):
                split += 1
            if split >= len(ids) - 5:
                continue  # no room left for a meaningful reference after adjusting
            prompt_ids = ids[:split]
            ref_ids = ids[split:split + CONTINUATION_TOKENS]
            ref_text = tokenizer.decode(ref_ids)
            if not ref_text.strip():
                continue
            input_ids = torch.tensor([prompt_ids], device=device)
            with torch.no_grad():
                generated = model.generate(input_ids, max_new_tokens=CONTINUATION_TOKENS, temperature=temp, greedy=False)
            gen_ids = generated[0].cpu().numpy().tolist()[len(prompt_ids):]
            gen_text = tokenizer.decode(gen_ids)
            hyps.append(gen_text if gen_text.strip() else " ")
            refs.append(ref_text)
            if len(examples) < 3:
                examples.append({"prompt": tokenizer.decode(prompt_ids), "reference": ref_text, "generated": gen_text})

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        rouge_l = sum(scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)) / len(hyps) if hyps else 0.0

        all_gen_tokens = []
        for h in hyps:
            all_gen_tokens.extend(tokenizer.encode(h).ids)
        distinct_1 = distinct_n(all_gen_tokens, 1)
        distinct_2 = distinct_n(all_gen_tokens, 2)
        rep_rate = repetition_rate(all_gen_tokens, 2)

        temp_results[str(temp)] = {
            "bleu4": round(bleu, 4), "chrf": round(chrf, 4), "rouge_l": round(rouge_l, 4),
            "distinct_1": round(distinct_1, 4), "distinct_2": round(distinct_2, 4),
            "repetition_rate_bigram": round(rep_rate, 4), "num_samples": len(hyps),
        }
        sample_generations[str(temp)] = examples
        print(f"  T={temp}: BLEU={bleu:.3f} chrF={chrf:.3f} ROUGE-L={rouge_l:.4f} "
              f"Distinct-1={distinct_1:.4f} Distinct-2={distinct_2:.4f} rep_rate={rep_rate:.4f} (n={len(hyps)})")

    return {
        "label": cfg["label"], "checkpoint": str(cfg["ckpt"].relative_to(ROOT)),
        "params": model.count_parameters(),
        "ppl_bpb": ppl_bpb, "generation": temp_results, "sample_generations": sample_generations,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for cfg in VARIANTS:
        results[cfg["key"]] = evaluate_variant(cfg, device)

    out_path = METRICS_DIR / "pretrained_eval_all4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {out_path.relative_to(ROOT)}")

    print("\nSummary:")
    print(f"{'model':<14} {'PPL':>10} {'BPB':>8} {'BLEU@1.0':>9} {'chrF@1.0':>9} {'ROUGE-L@1.0':>12} {'D-1@1.0':>8} {'D-2@1.0':>8}")
    for key, r in results.items():
        g = r["generation"]["1.0"]
        print(f"{key:<14} {r['ppl_bpb']['ppl']:>10.2f} {r['ppl_bpb']['bpb']:>8.4f} {g['bleu4']:>9.3f} {g['chrf']:>9.3f} {g['rouge_l']:>12.4f} {g['distinct_1']:>8.4f} {g['distinct_2']:>8.4f}")


if __name__ == "__main__":
    main()
