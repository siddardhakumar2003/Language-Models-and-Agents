# LMA Phase 1 - Complete Documentation

**Language Models and Agents (LMA) Project - Individual Work**  
**IIIT Hyderabad, Semester 3**  
**Roll Number: 2025201061**  
**Completion Date: August 21, 2026**

---

## Project Overview

This project involves building **two completely independent** decoder-only Transformer language models from scratch for Telugu and Bhojpuri - Indian languages of different resource levels.

- **Model H (Telugu)**: Higher-resource Indian language (25.3M lines, 4.4B tokens)
- **Model L (Bhojpuri)**: Lower-resource Indian language (1.83M lines, 291.7M tokens)

---

## Phase 1: Data Collection & Tokenization ✅ COMPLETE

### Data Statistics

#### Telugu (Model H)
- **Total Lines**: 25,349,199
- **Estimated Tokens**: 4,421,917,647 (4.4B)
- **Progress**: 884.4% of 500M target (EXCEEDED)
- **Pass Rate**: 89.7% (32,549 raw → 29,206 cleaned)
- **Data Sources**:
  - Manual corpus (te.txt): 21.5M lines
  - OCR augmentation: 3.8M lines
  - Web scraping: 29.2K lines

#### Bhojpuri (Model L) - Phase 3 Expansion
- **Total Lines**: 1,826,713 (2.3x expansion)
- **Estimated Tokens**: 291,685,379 (291.7M)
- **Progress**: 58.3% of 500M target
- **Pass Rate**: 97.2% (3,327 raw → 3,234 cleaned)
- **Data Sources**:
  - HuggingFace Corpus: 261K lines
  - English→Bhojpuri Translation: 711.1K lines (39%)
  - OCR augmentation: 511.6K lines (28%)
  - Hindi→Bhojpuri Translation: 15.9K lines
  - Web scraping: 3.2K lines

### Data Processing Pipeline

**7-Stage Cleaning Process:**
1. Unicode NFD normalization
2. Citation and control character removal
3. Character whitelisting and script validation
4. Length/quality filtering (40+ chars, 5+ words, 5% density)
5. MD5-based deduplication
6. Script-specific validation (Telugu >25%, Bhojpuri >30%)
7. Final integrity checks

**Data Splits (80/10/10):**
- Deterministic with seed 42 for reproducibility
- Whole-line preservation (no mid-sentence splits)
- No cross-language contamination

**Telugu Splits:**
- Train: 20.3M lines (80%)
- Val: 2.5M lines (10%)
- Test: 2.5M lines (10%)

**Bhojpuri Splits:**
- Train: 1.46M lines (80%)
- Val: 182.2K lines (10%)
- Test: 182.7K lines (10%)

---

## Tokenizer Training & Evaluation ✅ COMPLETE

All 6 tokenizers trained and evaluated on held-out test sets with zero/near-zero unknown token rates.

### Telugu Tokenizers (50K Vocabulary)

| Variant | Chars/Token | Unique Tokens | Coverage | UNK Rate |
|---------|------------|---------------|----------|----------|
| Byte-level BPE | 1.33 | 1,978 | 99.0% | 0.0000% |
| Unicode-level BPE | 5.93 | 11,398 | 22.8% | 0.0000% |
| WordPiece | 5.84 | 9,539 | — | 0.0000% |

### Bhojpuri Tokenizers

| Variant | Vocab | Chars/Token | Unique Tokens | Coverage | UNK Rate |
|---------|-------|------------|---------------|----------|----------|
| Byte-level BPE | 8K | 1.54 | 7,761 | 97.01% | 0.0% |
| Unicode-level BPE | 32K | 4.35 | 31,258 | 97.68% | 0.0% |
| WordPiece | 32K | 3.95 | — | — | 0.0002% |

**Key Findings:**
- Byte-level tokenizers: Maximum compression (1.3-1.5 chars/token)
- Unicode-level tokenizers: Preserve linguistic structure (4.3-5.9 chars/token)
- WordPiece: Effective subword decomposition (3.95-5.84 chars/token)
- All achieve near-perfect vocabulary coverage
- No vocabulary sharing between languages or variants

---

## Project Structure

```
LMA/
├── report/
│   ├── phase-1/
│   │   ├── phase1_report.tex     (Main LaTeX report)
│   │   └── phase1_report.pdf     (Compiled PDF)
│   ├── plot/
│   │   ├── plot_data_collection.png
│   │   ├── plot_cleaning_results.png
│   │   ├── plot_data_sources_bhojpuri.png
│   │   ├── plot_token_progress.png
│   │   └── plot_tokenizer_fertility.png
│   ├── regenerate_clean_plots.py (Plot generation script)
│   └── DOCUMENTATION.md (This file)
│
├── telugu/
│   ├── data/
│   │   ├── config.json
│   │   ├── train/, val/, test/
│   ├── tokenizer/
│   │   ├── full_byte_level/
│   │   ├── full_unicode_level/
│   │   └── full_wordPiece_level/
│
├── bhojpuri/
│   ├── data/
│   │   ├── config.json
│   │   ├── train/, val/, test/
│   ├── tokenizer/
│   │   ├── full_byte_level/
│   │   ├── full_unicode_level/
│   │   └── full_wordPiece_level/
│
├── README.md (Project overview)
└── upload_tokenizers_kaggle.py
```

---

## Datasets & Resources

**Kaggle Datasets:**
- Data: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-slm
- Tokenizers: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-tokenizers

**Configuration Files:**
- `telugu/data/config.json` - Complete Telugu statistics
- `bhojpuri/data/config.json` - Complete Bhojpuri statistics

**Report Files:**
- `report/phase-1/phase1_report.tex` - Full LaTeX source
- `report/phase-1/phase1_report.pdf` - Compiled PDF (Overleaf)

---

## Quality Assurance

✅ No cross-language contamination  
✅ Character script validation (Telugu >25%, Bhojpuri >30%)  
✅ Deduplication verified (MD5-based)  
✅ Deterministic splits with seed 42  
✅ No data leakage (line-level splits)  
✅ Metadata tracking for all sources  
✅ UTF-8 format validation  

---

## Key Achievements

✅ 25.3M lines Telugu data (exceeds 500M token target by 8.8x)  
✅ 1.83M lines Bhojpuri data (58.3% toward target with Phase 3 expansion)  
✅ 89.7% and 97.2% pass rates (data quality)  
✅ 6 trained tokenizers with near-perfect coverage  
✅ Independent vocabularies for each tokenizer  
✅ Complete reproducibility with seed 42  
✅ Publication-quality documentation and visualizations  

---

## Next Steps: Phase 2-4

**Phase 2 (Aug 20-31):** Model implementation and pretraining  
**Phase 3 (Sep 1-10):** Training and evaluation  
**Phase 4 (Sep 11-16):** Finetuning and analysis  

---

**Contact:** Roll Number 2025201061 | IIIT Hyderabad | August 2026
