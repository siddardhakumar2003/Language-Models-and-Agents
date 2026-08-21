# Report Code Directory

All Python scripts for plot/graph generation and dataset management for LMA Phase 1 report.

## Main Plot Generation Scripts

### `regenerate_clean_plots.py`
**Purpose**: Generate all 10 publication-quality plots for Phase 1 report  
**Usage**: `python3 regenerate_clean_plots.py`

**Output Plots**:
1. `plot_data_collection.png` - Telugu vs Bhojpuri lines and tokens
2. `plot_cleaning_results.png` - Data cleaning pass rates
3. `plot_data_splits.png` - 80/10/10 train/val/test splits
4. `plot_data_sources_bhojpuri.png` - Bhojpuri data source breakdown
5. `plot_token_progress.png` - Progress toward 500M token target
6. `plot_tokenizer_vocab.png` - Vocabulary efficiency
7. `plot_tokenizer_fertility.png` - Compression efficiency (all 6 tokenizers)
8. `plot_tokenizer_coverage.png` - Vocabulary coverage percentages
9. `plot_tokenizer_unk_rate.png` - Unknown token rates
10. `plot_tokenizer_analysis.png` - Fertility analysis

**Data Source**: Reads from `config.json` files  
**Output**: `../plot/` directory

---

## Legacy Plot Generation Scripts

### `generate_clean_charts.py`
**Purpose**: Earlier version of publication-quality chart generation  
**Status**: Use `regenerate_clean_plots.py` instead

### `generate_phase1_report.py`
**Purpose**: Complete report generation with embedded plots and statistics  
**Status**: Alternative report generator

### `generate_tokenizer_chart.py`
**Purpose**: Generate tokenizer comparison charts  
**Status**: Use `regenerate_clean_plots.py` instead

### `generate_tokenizer_clean.py`
**Purpose**: Clean tokenizer visualization generation  
**Status**: Legacy version

### `generate_fertility_only.py`
**Purpose**: Generate fertility (compression) plot only  
**Status**: Legacy, included in `regenerate_clean_plots.py`

### `generate_source_pie_chart.py` & `generate_source_pie_chart_v2.py`
**Purpose**: Generate Bhojpuri data source breakdown pie charts  
**Status**: Legacy versions, included in main generator

### `fix_bhojpuri_pie.py`
**Purpose**: Fix specific Bhojpuri pie chart formatting  
**Status**: Legacy fix script

### `evaluate_all_tokenizers.py`
**Purpose**: Evaluate all 6 tokenizers on test sets  
**Status**: Tokenizer evaluation utility

---

## Kaggle Integration

### `upload_tokenizers_kaggle.py`
**Purpose**: Prepare and upload all 6 trained tokenizers to Kaggle  
**Status**: Preparation complete (ready to upload)

**Prepares**:
- Directory structure with organized tokenizers
- Metadata JSON with evaluation metrics
- README with usage instructions
- All 6 tokenizer files (Telugu: 3, Bhojpuri: 3)

**Kaggle Dataset**: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-tokenizers

**Usage**:
```bash
export KAGGLE_API_TOKEN=<your_token>
python3 upload_tokenizers_kaggle.py
```

---

### `kaggle_tokenizer.py`
**Purpose**: Legacy tokenizer upload script (metadata and README generation)  
**Status**: Superseded by `upload_tokenizers_kaggle.py`

---

### `kaggle_data_upload.py`
**Purpose**: Upload train/val/test data splits to Kaggle  
**Status**: Preparation complete (ready to upload)

**Uploads**:
- Telugu data splits (train/val/test)
- Bhojpuri data splits (train/val/test)

**Kaggle Dataset**: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-slm

**Usage**:
```bash
export KAGGLE_API_TOKEN=<your_token>
python3 kaggle_data_upload.py
```

---

## Integration

These scripts are used to:
1. **Generate** publication-quality visualizations for the LaTeX report
2. **Prepare** datasets and tokenizers for Kaggle publication
3. **Upload** to Kaggle for community access and reproducibility

All plots are generated from the raw `config.json` files in the data directories, ensuring consistency between report statistics and visualizations.

---

## Output Directories

- **Plots**: `../plot/` (10 PNG files)
- **Report**: `../phase-1/` (LaTeX + PDF)
- **Documentation**: `../DOCUMENTATION.md`

---

**Last Updated**: 2026-08-21  
**Author**: Roll Number 2025201061  
**Project**: Language Models and Agents (LMA) - Phase 1
