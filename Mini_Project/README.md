# Language Models and Agents - Individual Project

**Building and Analyzing Monolingual Transformer LMs in Two Indian Languages**

**Languages**: Telugu (Model H - Higher-resource) and Bhojpuri (Model L - Lower-resource)

**Project Duration**: 12 Aug 2026 - 16 Sep 2026 (5 weeks)

---

## Project Overview

This project involves building **two completely independent** decoder-only Transformer language models from scratch:

- **Model H (Telugu)**: Higher-resource non-English Indian language
- **Model L (Bhojpuri)**: Lower-resource Indian language (from allowed list)

Each model has:
- ✓ Independent dataset (~500M tokens minimum per model)
- ✓ Independent tokenizer & vocabulary
- ✓ Independent model weights
- ✓ Separate preprocessing, training, evaluation pipelines

**Key Constraint**: Models are completely independent - NO shared data, tokenizer, or weights.

---

## Repository Structure

```
repo/
├── README.md                          # This file + reproduction steps
├── .gitignore
│
├── telugu/                            # Model H (Higher-resource)
│   ├── configs/
│   │   ├── tokenizer_config.json
│   │   ├── model_config.json
│   │   └── training_config.json
│   ├── tokenizer/
│   │   ├── train_tokenizer.py
│   │   └── telugu_tokenizer.json      # Trained tokenizer (Drive link)
│   ├── model/
│   │   ├── transformer.py             # Model implementation
│   │   └── __init__.py
│   ├── train/
│   │   ├── train.py
│   │   ├── dataset.py
│   │   └── checkpoint_latest.pt       # (Drive link)
│   ├── eval/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── results/
│   ├── data/
│   │   ├── raw/
│   │   │   └── telugu.txt             # Raw corpus (~500M tokens)
│   │   ├── processed/
│   │   │   └── train_val_test.txt
│   │   └── dataset_statistics.json
│   └── report/
│       ├── phase_1_report.md
│       ├── phase_2_report.md
│       ├── phase_3_final_report.md
│       └── visualizations/
│
├── bhojpuri/                          # Model L (Lower-resource)
│   ├── configs/
│   ├── tokenizer/
│   ├── model/
│   ├── train/
│   ├── eval/
│   ├── data/
│   └── report/
│
└── shared/                            (MINIMAL - only utility code)
    ├── utils.py
    ├── constants.py
    └── requirements.txt
```

---

## Phases and Deadlines

| Phase | Task | Duration | Deadline | Marks |
|-------|------|----------|----------|-------|
| 1 | Data Collection & Tokenizer Construction | 1 week | 19 Aug 2026 | 25 |
| 2 | Model Implementation, Pretraining, Evaluation | 2.5 weeks | 5 Sep 2026 | 40 |
| 3 | Reasoning Finetuning, Attention Analysis, Final Report | 1.5 weeks | 16 Sep 2026 | 35 |
| **Total** | | **5 weeks** | **16 Sep 2026** | **100** |

---

## Phase 1: Data Collection & Tokenizer Construction (19 Aug 2026)

### 1.1 Language Selection

- **Model H (Telugu)**: Higher-resource, abundant public text
- **Model L (Bhojpuri)**: Lower-resource (from: Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, Nepali, Sindhi)

### 1.2 Dataset Requirements

**Target**: ~500M tokens per language (minimum)

- At least **20% manual collection** (OCR from books/PDFs, typing/transcription)
- Remaining 80% from public sources (Wikipedia, Hugging Face, etc.)
- Report exact token counts and manual vs. downloaded split
- Maintain separate train/val/test splits

### 1.3 Tokenizer Training

- Train BPE tokenizer independently for each language
- Vocabulary size: ~10K-30K tokens
- Save tokenizer configuration and trained files
- Document tokenization statistics (token distribution, coverage, etc.)

### Deliverables (Phase 1)

```
telugu/
├── data/dataset_statistics.json       # Token counts, split info
├── data/raw/telugu.txt
├── data/processed/train.txt
├── data/processed/val.txt
├── data/processed/test.txt
├── tokenizer/telugu_tokenizer.json    # (Drive link if >50MB)
└── report/phase_1_report.md

bhojpuri/
└── [Same structure]
```

---

## Phase 2: Model Implementation, Pretraining, Evaluation (5 Sep 2026)

### 2.1 Model Architecture

- **Type**: Decoder-only Transformer
- **Size**: ~25M parameters per model
- **Architecture**: Basic PyTorch components (no HuggingFace Transformers)
- Implementation components:
  - Embedding layer
  - Multi-head self-attention
  - Feed-forward networks
  - Layer normalization
  - Positional encoding

### 2.2 Pretraining

- **Objective**: Next-token prediction
- **Optimizer**: Adam or similar
- **Learning rate schedule**: Warmup + cosine decay
- **Checkpointing**: Save intermediate checkpoints (mandatory for Colab recovery)
- **Training duration**: ~2-5 days per model

### 2.3 Evaluation

- **Metrics**: Perplexity, character-level BPE coverage
- **Evaluation set**: Held-out test set (10% of data)
- **Analysis**: 
  - Loss curves over training
  - Attention pattern visualization
  - Token-level accuracy

### Deliverables (Phase 2)

```
telugu/
├── model/transformer.py               # Model implementation
├── train/train.py                     # Training script
├── train/checkpoint_latest.pt         # (Drive link)
├── eval/results/metrics.json
├── eval/results/loss_curves.png
└── report/phase_2_report.md

bhojpuri/
└── [Same structure]
```

---

## Phase 3: Reasoning Finetuning & Analysis (16 Sep 2026 - FINAL)

### 3.1 Reasoning Tasks

- **Task 1**: Semantic similarity task (2-3 sentence pairs, classify as similar/dissimilar)
- **Task 2**: Basic reasoning (multiple choice QA on short passages)
- Create dataset: ~100-200 examples per model

### 3.2 Attention Analysis

- Visualize and analyze attention patterns
- Identify what linguistic phenomena models attend to
- Document findings

### 3.3 Final Report

- Complete project summary
- Reproduction steps (all hyperparameters, data splits, commands)
- Google Drive links for datasets and checkpoints
- Analysis of model behavior and limitations

### Deliverables (Phase 3)

```
telugu/
├── eval/finetune_reasoning.py
├── eval/results/reasoning_results.json
├── eval/attention_heatmaps/
├── report/phase_3_final_report.md
└── report/attention_analysis.md

bhojpuri/
└── [Same structure]
```

---

## Key Requirements

### Code Quality

1. ✓ **Modular design**: Separate data, model, training, evaluation code
2. ✓ **Documentation**: Docstrings for all methods
3. ✓ **Reproducibility**: Log hyperparameters with checkpoints
4. ✓ **Comments**: Explain non-obvious code sections
5. ✓ **Visualization**: Plot loss curves, attention patterns with proper labels

### Data Management

- ✓ Keep data separate per language
- ✓ NO concatenation of language corpora
- ✓ Track train/val/test splits
- ✓ Report dataset statistics for each phase
- ✓ Use Google Drive for large artifacts (>50MB)

### Model Independence

- ✓ Completely separate tokenizers
- ✓ Completely separate model weights
- ✓ Completely separate vocabularies
- ✓ No shared embeddings or layers
- ✓ Independent training runs

### Checkpointing (MANDATORY)

Save intermediate checkpoints including:
- Model weights
- Optimizer state
- Scheduler state
- Current epoch/step
- Training configuration

Required for recovery from Colab interruptions.

---

## Submission Policy

### Branches

- **phase-1**: Data collection & tokenizer work (Deadline: 19 Aug)
- **phase-2**: Model & training work (Deadline: 5 Sep)
- **phase-3**: Finetuning & analysis (Deadline: 16 Sep)

### Commit Policy

- Push frequently (daily commits expected)
- Meaningful commit messages
- No placeholder commits

### Large Artifacts

- ✓ Upload to Google Drive
- ✓ Include shareable links in README
- ✗ Do NOT commit binary files (>50MB) to git

### Per-Phase Reports

Each branch must include report/ folder with:
- Markdown/PDF report
- Plots and visualizations
- Heatmaps and analysis results
- Links to Google Drive data

---

## Implementation Guidelines

1. **Language-specific organization**: Each language directory is self-contained
2. **Clear module separation**: data → tokenizer → model → train → eval
3. **Visualization**: Use matplotlib/seaborn for all plots
4. **Reproducibility**: Include README with reproduction steps
5. **Documentation**: Complete docstrings and inline comments

---

## Getting Started

### Phase 1 Tasks

```bash
# Telugu setup
cd telugu/
python tokenizer/train_tokenizer.py
python train/dataset.py                    # Create splits
python eval/evaluate_tokenizer.py          # Analyze

# Bhojpuri setup
cd ../bhojpuri/
# Same steps...
```

### Phase 2 Tasks

```bash
cd telugu/
python train/train.py --config configs/training_config.json
python eval/evaluate.py
```

### Phase 3 Tasks

```bash
cd telugu/
python eval/finetune_reasoning.py
python eval/analyze_attention.py
```

---

## Resources

- **GitHub Classroom**: https://classroom.github.com/a/Q6g0Cxoh
- **Discussion Forum**: https://hackmd.io/@CL3410/ryyi0BUIGe
- **Deadline**: 16 September 2026, 11:59 P.M. IST

---

## Grading Criteria

- **Correctness**: Proper implementation of transformer architecture
- **Code Quality**: Clarity, modularity, documentation
- **Reproducibility**: Complete setup and training instructions
- **Analysis**: Meaningful insights from model behavior
- **Report**: Clear explanation of methodology and findings

Good luck! 🚀
