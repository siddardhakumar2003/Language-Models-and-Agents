# Language Models and Agents — Individual Project

**Building and Analyzing Monolingual Transformer LMs in Two Indian Languages**

**Languages**: Telugu (Model H — Higher-resource) and Bhojpuri (Model L — Lower-resource)

**Project Duration**: 12 Aug 2026 – 16 Sep 2026 (5 weeks)

**Code repository**: https://github.com/CL3-410/individual-project-siddardhakumar2003/tree/main/Mini_Project

---

## 📄 Final Report

The single, consolidated, LaTeX-compiled report covering all three phases plus the bonus
ablation is:

- **Source**: [`report/final_report.tex`](report/final_report.tex)
- **Compiled PDF**: [`report/final_report.pdf`](report/final_report.pdf) (9 pages)

It covers: Phase 1 data/tokenizer statistics, the Phase 2 architecture and all four pretrained
models' results, Phase 3 reasoning finetuning and attention re-analysis, the bonus
no-positional-embeddings ablation, a summary table of all four ablation studies run in this
project, and the four required Model H vs. Model L comparison questions. Every number in it is
measured directly from real training logs, real checkpoints, or real inference runs — no
synthetic/fabricated metrics.

Per-phase detailed reports (full temperature tables, every attention heatmap, complete logs):

- Phase 1: [`report/phase-1/phase1_report.pdf`](report/phase-1/phase1_report.pdf)
- Phase 2: [`report/phase-2/report.md`](report/phase-2/report.md)
- Phase 3: [`report/phase-3/report.md`](report/phase-3/report.md)

To recompile the final report locally (no LaTeX distribution needed — [tectonic](https://tectonic-typesetting.github.io/)
fetches packages on first run):
```bash
tectonic -X compile report/final_report.tex
```

---

## Project Overview

Two **completely independent** decoder-only Transformer language models, implemented from
scratch in PyTorch (no `nn.Transformer`, no pretrained models or tokenizers):

- **Model H (Telugu)**: higher-resource non-English Indian language
- **Model L (Bhojpuri)**: lower-resource Indian language (from the allowed list)

Each language has its own dataset, tokenizer, and weights — no sharing between them. Each
language also has **two** independently-trained pretrained variants:

- a **low-parameter** pair (7.34M params each, architecture-matched across languages for a clean
  resource comparison), and
- a separately-trained, larger **high-parameter** pair (25.54M Telugu / 15.08M Bhojpuri).

Both languages are finetuned on a self-generated synthetic comparative-reasoning dataset
(3 of the 4 pretrained checkpoints were finetuned — see Phase 3 below for why). A bonus ablation
retrains Bhojpuri with positional embeddings removed entirely.

---

## Ablation Studies (4, all in the final report)

| # | Ablation | What varied | Key finding |
|---|---|---|---|
| 1 | Tokenizer family & vocabulary size | Byte-level BPE / Unicode-level BPE / WordPiece at multiple vocab sizes | WordPiece chosen; fertility ranges 1.3–5.9 chars/token across families |
| 2 | Parameter count, pretraining | Low- vs. high-parameter architecture, both languages | Telugu: low-param wins 4.7× PPL (high-param run stalled). Bhojpuri: low/high nearly tied |
| 3 | Parameter count, finetuning | Telugu low- vs. high-parameter checkpoint, finetuned | Low-param finetune wins 3.5× accuracy — base-model PPL predicts finetuning outcome better than parameter count |
| 4 (bonus) | No positional embeddings | Positional-embedding table present vs. removed, Bhojpuri high-param | PPL degrades only mildly (+5.5%); last-layer per-head attention collapses to near-perfect uniformity; greedy generation ~40% more repetitive |

---

## Repository Structure

```
Mini_Project/
├── README.md                          # This file
├── LMA_Individual_Project_v1.pdf      # Assignment spec
├── report/
│   ├── final_report.tex / .pdf        # ★ Primary consolidated submission report
│   ├── final_figs/                    # Figures embedded in final_report.pdf
│   ├── phase-1/                       # Phase 1 report + plots
│   ├── phase-2/                       # Phase 2 report + real eval/plot generation code
│   ├── phase-3/                       # Phase 3 report + real eval/plot generation code
│   └── code/                          # Tokenizer evaluation / Kaggle upload helpers
│
├── telugu/                            # Model H (Telugu)
│   ├── data/{train,val,test}/telugu.txt, fine_tune.txt
│   ├── data_collect/                  # Scraping, cleaning, OCR, translation scripts
│   ├── tokenizer/                     # WordPiece tokenizer training + trained files
│   ├── model/                         # transformer.py, transformer_no_pos.py (N/A here)
│   ├── model/outputs/                 # checkpoints/ (high), submission/ (low), finetune_checkpoints{,_low}/
│   ├── train/                         # train.py, pretrain_kaggle.ipynb, kaggle_bundle/
│   ├── finetune/                      # generate_reasoning_data.py, finetune.py, finetune_kaggle.ipynb
│   └── configs/                       # model/training/tokenizer/finetune configs (incl. *_low variants)
│
├── bhojpuri/                          # Model L (Bhojpuri)
│   ├── data/{train,val,test}/bhoj.txt, fine_tune.txt
│   ├── data_collect/                  # Scraping, cleaning, OCR, translation scripts
│   ├── tokenizer/                     # WordPiece tokenizer training + trained files
│   ├── model/
│   │   ├── transformer.py             # Standard BhojpuriTransformer
│   │   └── transformer_no_pos.py      # BhojpuriTransformerNoPos (bonus ablation)
│   ├── model/outputs/                 # checkpoints/, checkpoints_no_pos/, finetune_checkpoints/, submission_phase-2/
│   ├── train/
│   │   ├── train.py, pretrain_kaggle.ipynb, kaggle_bundle/
│   │   ├── train_no_pos.py, pretrain_no_pos_kaggle.ipynb   # bonus ablation training
│   └── finetune/                      # generate_reasoning_data.py, finetune.py, finetune_kaggle.ipynb
│
├── upload_checkpoints_to_kaggle.py    # -> kspsvln/checkpoint-{telugu,bhojpuri}
├── upload_data_to_kaggle.py           # -> kspsvlnsiddardha/lma-slm
├── upload_phase2_bundles.py           # -> kspsvln/lma-{telugu,bhojpuri}-phase2 (kaggle_bundle code+config)
└── report/code/upload_tokenizers_kaggle.py  # -> kspsvlnsiddardha/lma-tokenizers
```

---

## Phase 1: Data Collection & Tokenizer Construction

**Collection** — Telugu's Phase 1 collection totaled 25.35M lines (21.46M-line manual corpus +
web scraping + OCR), but **only a separately-prepared, smaller 3.89M-line corpus (509.3M
tokens) was actually used for Phase 2 pretraining** — the large manual corpus was collected but
never carried into later phases. Bhojpuri collected 1.83M lines (HuggingFace corpus + OCR +
Hindi/English→Bhojpuri NLLB-200 translation + web scraping, 291.7M tokens), and **all of it was
used** for pretraining (its train/val/test files match the collected total exactly).

**Cleaning** — identical 7-stage pipeline for both languages (Unicode NFD normalization,
citation/control-character removal, character whitelisting, script validation,
length/word-count/density filters, MD5 deduplication), deterministic 80/10/10 split (seed 42).

**Tokenizers** — 3 families trained and compared per language at exploratory sizes (byte-level
BPE, Unicode-level BPE, WordPiece); WordPiece selected for its explicit `##` subword-continuation
marking. The 4 pretrained models each use their own smaller, model-specific WordPiece vocabulary:
10,000 (Telugu/Bhojpuri low-parameter pair), 20,000 (Telugu high-parameter), 16,000 (Bhojpuri
high-parameter).

Full detail: [`report/phase-1/phase1_report.pdf`](report/phase-1/phase1_report.pdf).

---

## Phase 2: Model Implementation, Pretraining, and Evaluation

Decoder-only, GPT-style Transformer implemented entirely from primitives (`nn.Linear`,
`nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` only): manual multi-head causal self-attention,
learned absolute positional embeddings, pre-norm residuals, GELU feedforward, tied
input/output embeddings.

| | Telugu low | Telugu high | Bhojpuri low | Bhojpuri high |
|---|---|---|---|---|
| Vocab / $d_{model}$ / layers / heads | 10K / 256 / 6 / 8 | 20K / 384 / 10 / 8 | 10K / 256 / 6 / 8 | 16K / 320 / 8 / 8 |
| Parameters | 7.34M | 25.54M | 7.34M | 15.08M |
| Epochs completed | 16 (converged) | 36 (**stalled**) | 10 (converged) | 44 (converged) |
| **Best val. PPL** | **881.90** | **4119.81** | **870.14** | **814.48** |

The Telugu high-parameter run stalled mid-training (train loss frozen for 20 consecutive
epochs) — this is an engineering/scheduling artifact, not a resource-tier effect: on the
architecture-matched (low-parameter) pair, Bhojpuri (the lower-resource language) actually
edges out Telugu.

Full evaluation (PPL/BPB, BLEU-4/chrF/ROUGE-L at 3 temperatures, diversity/repetition,
real per-head attention heatmaps and entropy/distance for all 4 models) is in
[`report/final_report.pdf`](report/final_report.pdf) Sec. 3 and
[`report/phase-2/report.md`](report/phase-2/report.md).

---

## Phase 3: Reasoning Finetuning, Attention Re-Analysis, and Bonus Ablation

**Which checkpoints were finetuned**: both languages' high-parameter checkpoint, plus Telugu's
low-parameter checkpoint (Ablation Study 3) — **3 finetuned models total, not 4**. Bhojpuri's
low-parameter checkpoint was deliberately not finetuned: its pretrained PPL (870.1) is nearly
identical to Bhojpuri high-parameter's (814.5), so a low-vs-high finetuning ablation there
wouldn't isolate anything new, unlike Telugu's pair (4.7× PPL gap).

**Reasoning dataset** (`{telugu,bhojpuri}/finetune/generate_reasoning_data.py`): 20,000-example
synthetic comparative-reasoning QA set (16,000/2,000/2,000 train/val/test), 64 templates over 5
question types × 4 attributes (height/weight/age/price), entity-name and phrasing leakage
avoidance between splits.

| Model | Pretrained acc. | Finetuned overall acc. | Yes/no-type bucket | Name/value-type bucket |
|---|---|---|---|---|
| Telugu high-param | 0.0% | 2.50% | 3.31% | 0.00% |
| Telugu low-param | 0.0% | 8.75% | 10.91% | 0.00% |
| Bhojpuri high-param | 0.0% | 16.40% | 45.30% | 0.00% |

All three finetuned models learn yes/no-shaped answers reasonably well but score **exactly 0%**
on every category requiring the model to output the correct compared entity's *name* — a real,
specific failure mode only visible by breaking accuracy down per question type.

**Attention re-analysis** (`report/phase-3/code/attention_analysis.py`): real per-head attention
extraction on pretrained vs. finetuned checkpoints, early vs. late layer, both languages —
Bhojpuri's layer 1 shows a large, real entropy shift after finetuning (2.202→1.642), correlating
with it being the model that also improved most behaviorally; Telugu barely shifts at all.

**Bonus ablation — no positional embeddings** (Bhojpuri): `BhojpuriTransformerNoPos`
(`bhojpuri/model/transformer_no_pos.py`) subclasses the standard model, removing only the
positional-embedding table. Trained via `bhojpuri/train/train_no_pos.py` /
`pretrain_no_pos_kaggle.ipynb` — new files only, no existing code modified.

| | Standard (44 epochs) | No positional embeddings (15 epochs) |
|---|---|---|
| Full test-set PPL / BPB | 789.04 / 9.624 | 832.37 / 9.70 |
| Last-layer entropy, per head | 2.42–2.52 (real variation) | 2.3818–2.3821 (near-perfectly flat) |
| Greedy repetition rate | 0.331 | 0.462 |

PPL degrades only mildly (+5.5%) despite zero position information — causal masking alone gives
an implicit notion of sequence order — but per-head specialization collapses almost entirely,
and greedy-decoding repetition rises ~40%.

Full detail: [`report/final_report.pdf`](report/final_report.pdf) Secs. 4–6,
[`report/phase-3/report.md`](report/phase-3/report.md).

---

## Reproducibility

**Kaggle datasets** (checkpoints, tokenizers, raw data, code bundles — dataset IDs from the
upload scripts' own configuration; not independently re-verified live on Kaggle):

| What | Kaggle dataset |
|---|---|
| Telugu checkpoints | [kspsvln/checkpoint-telugu](https://www.kaggle.com/datasets/kspsvln/checkpoint-telugu) |
| Bhojpuri checkpoints | [kspsvln/checkpoint-bhojpuri](https://www.kaggle.com/datasets/kspsvln/checkpoint-bhojpuri) |
| Tokenizers (both languages, all variants) | [kspsvlnsiddardha/lma-tokenizers](https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-tokenizers) |
| Raw pretraining + finetuning corpus | [kspsvlnsiddardha/lma-slm](https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-slm) |
| Code+config bundles used on Kaggle | [kspsvln/lma-telugu-phase2](https://www.kaggle.com/datasets/kspsvln/lma-telugu-phase2), [kspsvln/lma-bhojpuri-phase2](https://www.kaggle.com/datasets/kspsvln/lma-bhojpuri-phase2) |

**Reproduction steps:**
```bash
# 1. Generate the reasoning finetuning dataset (both languages)
python3 telugu/finetune/generate_reasoning_data.py --num-samples 20000 --seed 42
python3 bhojpuri/finetune/generate_reasoning_data.py --num-samples 20000 --seed 42

# 2. Pretrain on Kaggle: {telugu,bhojpuri}/train/pretrain_kaggle.ipynb
#    (ROOT_DIR -> that language's lma-*-phase2 kaggle_bundle dataset)

# 3. Bonus ablation pretrain (Bhojpuri only): bhojpuri/train/pretrain_no_pos_kaggle.ipynb
#    (re-upload kaggle_bundle including the new train_no_pos.py / transformer_no_pos.py files first)

# 4. Finetune on Kaggle: {telugu,bhojpuri}/finetune/finetune_kaggle.ipynb
#    (PRETRAINED_CKPT -> that language's Phase 2 checkpoint; telugu_finetune_low.ipynb for
#    the low-parameter Telugu ablation)

# 5. Real attention analysis (pretrained vs. finetuned, both languages, runs on CPU)
python3 report/phase-3/code/attention_analysis.py --language both

# 6. Compile the final report
tectonic -X compile report/final_report.tex
```

**Load a trained model:**
```python
import torch
from telugu.model.transformer import TeluguTransformer
from bhojpuri.model.transformer import BhojpuriTransformer

telugu_model = TeluguTransformer()
ckpt = torch.load("telugu/model/outputs/checkpoints/checkpoint_best.pt", map_location="cpu")
telugu_model.load_state_dict(ckpt["model_state_dict"])

bhojpuri_model = BhojpuriTransformer()
ckpt = torch.load("bhojpuri/model/outputs/checkpoints/checkpoint_best.pt", map_location="cpu")
bhojpuri_model.load_state_dict(ckpt["model_state_dict"])
```

**Key constraints honored throughout**: no pretrained models/tokenizers, no `nn.Transformer`,
two fully independent monolingual LMs (no shared data/tokenizer/weights), positional embeddings
+ multi-head attention + causal masking are all student-implemented from primitives, seed 42
fixed for all random operations.
