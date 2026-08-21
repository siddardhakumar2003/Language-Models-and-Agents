# Language Models and Agents - Individual Project

**Building and Analyzing Monolingual Transformer LMs in Two Indian Languages**

**Languages**: Telugu (Model H - Higher-resource) and Bhojpuri (Model L - Lower-resource)

**Project Duration**: 12 Aug 2026 - 16 Sep 2026 (5 weeks)

---

## 📊 Progress Summary

### Phase 1: Data Collection & Tokenizer Construction ✅ **COMPLETE**

**Status**: All data collected, cleaned, split, and ready for tokenization (Deadline: 19 Aug 2026)

#### Telugu (Model H)
- ✅ **Total Data**: 25,349,199 lines (25.3M)
  - Fresh scraped: 29,206 lines (0.1%) from 6 web sources
  - Manual (te.txt): 21,458,261 lines (84.6%) existing corpus
  - OCR augmentation: 3,766,703 lines (14.9%) from archive.org
- ✅ **Estimated Tokens**: 4,421,917,647 (4.4B tokens - 884.4% of 500M target)
- ✅ **Data Quality**: 89.7% pass rate (32,549 raw → 29,206 cleaned)
- ✅ **Train/Val/Test Splits**:
  - Train: 20,281,561 lines (80%, 14.0 GB)
  - Val: 2,534,020 lines (10%, 1.7 GB)
  - Test: 2,533,618 lines (10%, 1.7 GB)
- ✅ **Cleaning**: 7-stage pipeline (Unicode normalization, deduplication, validation)
- ✅ **Configuration**: telugu/data/config.json with complete statistics

#### Bhojpuri (Model L)
- ✅ **Total Data**: 1,826,713 lines (2.3x expansion from Phase 1)
  - HuggingFace corpus (Satyam810/BhojpuriCorpus): ~261K lines (primary source)
  - OCR augmentation: 511,599 lines (28.0%) from archive.org pre-OCR'd texts
  - English→Bhojpuri Translation (fineweb-edu): 711,138 lines (39.0%) via NLLB-200
  - Hindi→Bhojpuri Translation: 15,852 lines via NLLB-200
  - Web scraping: 3,234 lines from Hindi/Bhojpuri news sites
- ✅ **Estimated Tokens**: 291,685,379 (291.7M tokens - 58.3% of 500M target) ⬆️ from 27.3%
- ✅ **Corpus Size**: 1,112.7 MB
- ✅ **Scraping Quality**: 97.2% pass rate (3,327 raw → 3,234 cleaned)
- ✅ **Train/Val/Test Splits**:
  - Train: 1,461,821 lines (80%)
  - Val: 182,219 lines (10%)
  - Test: 182,673 lines (10%)
- ✅ **Cleaning**: 7-stage pipeline with Devanagari script validation + deduplication
- ✅ **Configuration**: bhojpuri/data/config.json with complete statistics (updated 2026-08-21)

#### Key Achievements
- ✅ API-driven web scraping (MediaWiki API for automatic article discovery)
- ✅ Checksummed duplicate removal (MD5 hash-based per-batch tracking)
- ✅ Unicode normalization (NFD - Canonical Decomposition)
- ✅ Language-specific script validation (Telugu: >25%, Bhojpuri: >30%)
- ✅ No cross-language contamination (separate pipelines, independent data)
- ✅ Deterministic 80/10/10 splits with seed 42 (reproducible)
- ✅ Complete documentation in PROJECT_SUMMARY.md

### Phase 1b: Tokenizer Training ✅ **COMPLETE**
- ✅ **Telugu Tokenizers** (3 variants, 50K vocab each):
  - Byte-Level BPE: 1,978 unique tokens used, 1.33 chars/token, 0.0% UNK rate
  - Unicode-Level BPE: 11,398 unique tokens used, 5.93 chars/token, 0.0% UNK rate
  - WordPiece: 9,539 unique tokens used, 5.84 chars/token, 0.0% UNK rate
- ✅ **Bhojpuri Tokenizers** (3 variants):
  - Byte-Level BPE (8K vocab): 7,761 unique tokens, 1.54 chars/token, 97.01% coverage, 0.0% UNK rate
  - Unicode-Level BPE (32K vocab): 31,258 unique tokens, 4.348 chars/token, 97.68% coverage, 0.0% UNK rate
  - WordPiece (32K vocab): 11.4M test tokens, 0.0002% UNK rate
- ✅ All tokenizers evaluated on held-out test sets with zero/near-zero unknown rates
- ✅ Complete tokenizer evaluation report with fertility analysis and coverage metrics

### Phase 2: Model Implementation, Pretraining, Evaluation 🔄 **PENDING**
- ⏳ Transformer model implementation (decoder-only)
- ⏳ Pretraining with next-token prediction
- ⏳ Evaluation and metrics
- ⏳ Checkpoint saving and model weights

### Phase 3: Reasoning Finetuning & Analysis 🔄 **PENDING**
- ⏳ Reasoning task dataset creation
- ⏳ Finetuning on semantic similarity + QA tasks
- ⏳ Attention pattern analysis
- ⏳ Final comprehensive report

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

## Phase 1: Data Collection & Tokenizer Construction ✅ COMPLETE

### 1.1 Language Selection ✅

- **Model H (Telugu)**: Higher-resource, abundant public text ✅
- **Model L (Bhojpuri)**: Lower-resource (Devanagari proxy - Hindi Wikipedia) ✅

### 1.2 Dataset Completion ✅

**Telugu**: 25,349,199 lines total
- **Manual Corpus**: 21,458,261 lines (84.6%) - te.txt existing ✅
- **Web Scraped**: 29,206 lines (0.1%) - 6 sources ✅
- **OCR Augmentation**: 3,766,703 lines (14.9%) - archive.org ✅
- **Token Estimate**: 4,421,917,647 (4.4B tokens) - 884.4% of 500M target ✅

**Bhojpuri**: 776,801 lines (multi-source)
- **Web Scraping**: 3,234 lines (0.4%) - 8 Hindi/Bhojpuri sites ✅
- **HuggingFace Corpus**: ~261K lines (33.6%) - Satyam810/BhojpuriCorpus primary source ✅
- **OCR Augmentation**: 511,599 lines (65.9%) - archive.org pre-OCR'd texts ✅
- **Hindi→Bhojpuri MT**: 15,852 lines via NLLB-200 ✅
- **Wikipedia**: ~8,900 articles - included in HF deduplicated corpus
- **Quality**: 97.2% pass rate on scraped data ✅
- **Token Estimate**: 136,606,695 (136.6M tokens) - 27.3% of 500M target

### 1.3 Data Sources ✅

**Telugu Web Sources**:
1. te.wikipedia.org (3,000 articles via MediaWiki API) ✅
2. te.wikibooks.org (500+ pages) ✅
3. te.wikiquote.org (300+ quotes) ✅
4. te.wikisource.org (300+ texts) ✅
5. eenadu.net (Telugu news) ✅
6. greatandhra.com (Telugu news/content) ✅

**Bhojpuri Web Sources** (Hindi Devanagari proxy):
1. hi.wikipedia.org (800 articles via API) ✅
2. BBC Hindi ✅
3. Aajtak ✅
4. NDTV ✅
5. Hindustan Times ✅
6. Webdunia Hindi ✅
7. Bhaskar ✅
8. Other Hindi content sites ✅

### 1.4 Data Cleaning Pipeline ✅

**7-Stage Cleaning Process** (both languages):
1. Unicode NFD normalization ✅
2. Citation removal ✅
3. Control character removal ✅
4. Character whitelisting ✅
5. Script-specific validation ✅
6. Length/words/density filters ✅
7. Deduplication (MD5 hashes) ✅

**Results**:
- Telugu: 32,549 raw → 29,206 cleaned (89.7% pass rate, 181 duplicates removed) ✅
- Bhojpuri: 3,327 raw → 3,234 cleaned (97.2% pass rate, 11 duplicates removed) ✅

### 1.5 Train/Val/Test Splits ✅

**Split Method**: 80/10/10 with seed 42 (deterministic, reproducible)
- **In-memory shuffle**: Small files (scraped data)
- **Streaming order-preserved**: Large files (15GB te.txt)
- **Whole-line only**: No mid-line splits

**Telugu Splits**:
```
train/: telugu.txt (23,364) + te.txt (17,168,831) = 17,192,195 lines ✅
val/:   telugu.txt (2,920) + te.txt (2,144,950) = 2,147,870 lines ✅
test/:  telugu.txt (2,922) + te.txt (2,144,480) = 2,147,402 lines ✅
```

**Bhojpuri Splits**:
```
train/: 2,587 lines (80%) ✅
val/:   323 lines (10%) ✅
test/:  324 lines (10%) ✅
```

### Deliverables (Phase 1) ✅

```
report/phase-1/
├── phase1_report.tex                  ✅ Comprehensive LaTeX report
├── phase1_report.pdf                  ✅ Compiled (use XeLaTeX/LuaLaTeX)
├── plot_data_collection.png           ✅ Collection metrics
├── plot_cleaning_results.png          ✅ Cleaning pipeline results
├── plot_data_splits.png               ✅ Train/Val/Test distribution
├── plot_data_sources_bhojpuri.png     ✅ Bhojpuri source composition
└── plot_token_progress.png            ✅ Token progress tracking

telugu/data/
├── config.json                        ✅ Full statistics & metadata
├── scrape_state.json                  ✅ Checkpoint
├── train/                              ✅ 20.3M lines (14 GB)
├── val/                                ✅ 2.5M lines (1.7 GB)
└── test/                               ✅ 2.5M lines (1.7 GB)

telugu/tokenizer/
├── full_byte_level/telugu_tokenizer_full.json        ✅ 50K vocab
├── full_unicode_level/checkpoint_batch_140.json      ✅ 50K vocab
└── full_wordPiece_level/telugu_wp_tokenizer.json     ✅ 50K vocab

bhojpuri/data/
├── config.json                        ✅ Full statistics (updated 2026-08-19)
├── scrape_state.json                  ✅ Checkpoint
├── train/                              ✅ 627.9K lines (414 MB)
├── val/                                ✅ 78.1K lines (52 MB)
└── test/                               ✅ 78.4K lines (52 MB)

bhojpuri/tokenizer/
├── full_byte_level/bhoj_tokenizer_report_full.json       ✅ 8K vocab
├── full_unicode_level/bhoj_tokenizer_report_full.json    ✅ 16K vocab
└── full_wordPiece_level/bhojpuri_wp_tokenizer.json       ✅ 16K vocab

README.md                                             ✅ Complete documentation
```

---

## Phase 1 Completion Status (2026-08-19)

### Dataset Summary
**Telugu (Model H)**:
- **Total Tokens**: 4,421,917,647 (4.4B) - 884.4% of 500M target ✅ EXCEEDED
- **Total Lines**: 25,349,199 (25.3M)
- **Corpus Size**: 17.4 GB
- **Status**: ✅ Ready for Phase 2

**Bhojpuri (Model L)** - MAJOR EXPANSION:
- **Total Tokens**: 291,685,379 (291.7M) - 58.3% of 500M target ⬆️ from 27.3%
- **Total Lines**: 1,826,713 (2.3x increase) ⬆️ from 776,801
- **Corpus Size**: 1,112.7 MB ⬆️ from 518 MB
- **New Sources**: English→Bhojpuri translation (711K lines) + Hindi→Bhojpuri translation (16K lines)
- **Status**: ✅ Ready for Phase 2 (substantial phase 3 expansion completed)

### Tokenizer Summary
- ✅ **6 Total Tokenizers Trained** (3 per language)
- ✅ **All Tokenizers Evaluated** on held-out test sets
- ✅ **Zero/Near-Zero UNK Rates** (0.0000% to 0.0001%)
- ✅ **Independent Vocabularies** (no vocabulary sharing)
- ✅ **Fertility Analysis** (chars per token: 1.33-5.93 for Telugu, 1.54-3.44 for Bhojpuri)

### Phase 1 Deliverables
- ✅ **Phase 1 Report**: `report/phase-1/phase1_report.tex` (full LaTeX document)
- ✅ **Data Plots**: 5 publication-quality visualizations (PNG)
- ✅ **Configuration Files**: Complete statistics in config.json files
- ✅ **Tokenizer Checkpoints**: All 6 trained tokenizers saved
- ✅ **Kaggle Dataset**: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-slm
- ✅ **Kaggle Tokenizers**: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-tokenizers

---

## Quick Start Guides

### Phase 1: View the Report
```bash
# Compile LaTeX report to PDF (requires XeLaTeX or LuaLaTeX)
cd report/phase-1/
xelatex phase1_report.tex
# or use Overleaf: https://www.overleaf.com/
# Copy phase1_report.tex and PNG plots to Overleaf, set compiler to XeLaTeX
```

### Phase 1: Access the Dataset & Tokenizers
- **Kaggle Dataset**: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-slm
- **Kaggle Tokenizers**: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-tokenizers
- **Local Paths**:
  - Telugu Data: `/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/telugu/data/`
  - Bhojpuri Data: `/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/bhojpuri/data/`
  - Telugu Tokenizers: `/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/telugu/tokenizer/`
  - Bhojpuri Tokenizers: `/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/bhojpuri/tokenizer/`

### Phase 1: Load Tokenizers
```python
from tokenizers import Tokenizer

# Telugu tokenizers
te_byte = Tokenizer.from_file("telugu/tokenizer/full_byte_level/telugu_tokenizer_full.json")
te_unicode = Tokenizer.from_file("telugu/tokenizer/full_unicode_level/checkpoint_batch_140.json")
te_wp = Tokenizer.from_file("telugu/tokenizer/full_wordPiece_level/telugu_wp_tokenizer.json")

# Bhojpuri tokenizers
bh_byte = Tokenizer.from_file("bhojpuri/tokenizer/full_byte_level/bhoj_tokenizer_report_full.json")
bh_unicode = Tokenizer.from_file("bhojpuri/tokenizer/full_unicode_level/bhoj_tokenizer_report_full.json")
bh_wp = Tokenizer.from_file("bhojpuri/tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json")
```

### Phase 2: Model Training (Upcoming)
See `telugu/train/` or `bhojpuri/train/` directories for training scripts (Phase 2).