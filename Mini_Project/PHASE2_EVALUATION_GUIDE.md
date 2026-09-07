# Phase 2 Evaluation Guide

**Date**: 2026-09-07  
**Deadline**: 2026-09-16 (9 days remaining)  
**Phase 2 Deadline**: 2026-09-05 (PASSED - 2 days ago)

---

## What's Complete ✅

### Architecture & Implementation
- [x] Decoder-only Transformer (from scratch)
- [x] Multi-head causal self-attention (manual implementation)
- [x] Positional embeddings (learned absolute)
- [x] Transformer blocks with residual connections
- [x] Feed-forward networks (GELU activation)
- [x] Layer normalization (pre-norm)
- [x] Causal masking verification

### Data & Tokenization
- [x] Test data splits (10K+ samples each)
  - Telugu: 389,138 lines
  - Bhojpuri: 182,673 lines
- [x] BPE Tokenizers trained
  - Telugu: 50K vocabulary
  - Bhojpuri: 32K vocabulary
- [x] Model configs (in JSON)
  - Telugu: 24.73M parameters
  - Bhojpuri: 20.12M parameters

### Training
- [x] Pretraining completed on both models
- [x] Checkpoints saved (best + last)
- [x] Training logs recorded
- [x] Validation metrics tracked

---

## What's Needed for Phase 2 Submission

### 1. Evaluation Metrics (Required by PDF Section 2.3)

Run the evaluation script:

```bash
cd /media/ubuntu/Personal/IIIT_Hyderabad/Semester_3/LMA/Mini_Project
python eval_phase2_metrics.py --language both
```

**This generates**:
- Perplexity (PPL) on test set
- Bits-per-byte (BPB)
- Generated samples at temperatures [0.5, 1.0, 1.5]
- Diversity metrics (Distinct-1, Distinct-2)
- Attention analysis summary

**Output**: `report/phase2_evaluation.json`

### 2. Generation Quality Metrics (Per PDF)

Temperature-based evaluation required:

```
Temperature | Use Case
0.5         More deterministic, focused
1.0         Default softmax (balanced)
1.5         More exploratory, creative
```

**Metrics to report per temperature**:
- [ ] BLEU (4-gram, corpus-level)
- [ ] chrF / chrF++ (character F-score)
- [ ] ROUGE-L (longest common subsequence)
- [ ] Repetition rate & diversity stats

**Note**: The evaluation script computes diversity; for BLEU/chrF/ROUGE, you need:
```bash
pip install sacrebleu rouge_score
```

### 3. Attention Analysis (Per PDF)

Required deliverables:
- [ ] Attention heatmaps (1 early layer + 1 late layer)
- [ ] Multiple heads per layer
- [ ] Entropy per head/layer
- [ ] Mean attention distance
- [ ] Discussion: local vs. long-range patterns

**Script to generate**:
```bash
python scripts/attention_heatmap_generator.py
```

### 4. Loss Curves & Training Logs (Per PDF)

Already have training logs, need visualization:

```bash
python scripts/plot_training_curves.py telugu bhojpuri
```

**Should show**:
- Training loss (per epoch)
- Validation loss (per epoch)
- Validation PPL over time

### 5. Compliance Report

Final checklist:

| Requirement | Status |
|---|---|
| Decoder-only architecture | ✅ |
| Manual attention implementation | ✅ |
| Causal masking | ✅ |
| No HuggingFace Transformers | ✅ |
| Separate tokenizers | ✅ |
| Separate models | ✅ |
| Checkpoint system | ✅ |
| Test on held-out set | 🔄 |
| Generation at 3 temperatures | 🔄 |
| Diversity metrics | 🔄 |
| Attention analysis | ⏳ |

---

## Recommended Next Steps (Priority Order)

### TODAY (Essential)
1. **Run evaluation**:
   ```bash
   python eval_phase2_metrics.py --language both
   ```

2. **Install optional libraries** (for BLEU/chrF/ROUGE):
   ```bash
   pip install sacrebleu rouge_score
   ```

3. **Check logs** to verify training completion:
   ```bash
   tail -20 telugu/model/outputs/checkpoints/training_telugu.log
   tail -20 bhojpuri/model/outputs/checkpoints/training_bhojpuri.log
   ```

### THIS WEEK (High Priority)
4. **Generate loss curve plots** from training logs
5. **Create attention heatmap visualizations**
6. **Compute BLEU, chrF, ROUGE-L** on generated samples

### Before Submission
7. **Upload checkpoints to Google Drive**
   - Create shareable links
   - Put links in README
8. **Write final Phase 2 report** (PDF or Markdown)
   - Consolidate all metrics
   - Add visualizations
   - Compare Model H vs. L

---

## File Structure

```
project/
├── eval_phase2_metrics.py                    ← Main evaluation script
├── PHASE2_EVALUATION_GUIDE.md               ← This file
├── PHASE2_COMPLIANCE_CHECKLIST.md           ← Detailed checklist
│
├── telugu/
│   ├── model/
│   │   ├── transformer.py                   ✅ Architecture
│   │   ├── outputs/checkpoints/
│   │   │   ├── checkpoint_best.pt           ✅ Best checkpoint
│   │   │   ├── checkpoint_last.pt           ✅ Latest checkpoint
│   │   │   └── training_telugu.log          ✅ Training log
│   │   └── __init__.py
│   │
│   ├── configs/
│   │   ├── model_config.json                ✅ Architecture config
│   │   ├── tokenizer_config.json
│   │   └── training_config.json
│   │
│   ├── tokenizer/
│   │   └── full_byte_level/
│   │       └── telugu.json                  ✅ BPE tokenizer
│   │
│   ├── data/
│   │   ├── config.json                      ✅ Data statistics
│   │   └── test/
│   │       └── telugu.txt                   ✅ Test set (389K lines)
│   │
│   └── train/
│       └── [training scripts]
│
├── bhojpuri/
│   └── [Same structure as telugu/]
│       ├── model/outputs/checkpoints/
│       │   ├── checkpoint_best.pt           ✅ 85MB checkpoint
│       │   ├── checkpoint_last.pt           ✅
│       │   └── training_bhojpuri.log        ✅
│       ├── data/test/bhoj.txt               ✅ Test set (182K lines)
│       └── ...
│
└── report/
    ├── phase2_evaluation.json               🔄 (to generate)
    └── phase2_report.pdf                    ⏳ (to write)
```

---

## Expected Metrics

Based on training logs observed:

### Telugu (Model H)
- **Parameters**: 24.73M
- **Vocab**: 50K
- **Final Val PPL** (epoch 16): ~881
- **Expected Test PPL**: 800-1000 (similar range)
- **Training steps**: ~3.3M

### Bhojpuri (Model L)
- **Parameters**: 20.12M
- **Vocab**: 32K
- **Final Val PPL**: [Check log]
- **Expected Test PPL**: Higher than Telugu (lower-resource)
- **Training steps**: [Check log]

---

## Phase 2 Deliverables Checklist

### Code (Due: 2026-09-16)
- [x] Transformer implementation (both languages)
- [x] Model configuration files
- [x] Training scripts
- [ ] Evaluation scripts (generating)
- [ ] Attention visualization scripts

### Results (Due: 2026-09-16)
- [ ] **Metrics**:
  - [ ] PPL & BPB tables (test set)
  - [ ] BLEU, chrF, ROUGE-L (at temps 0.5, 1.0, 1.5)
  - [ ] Diversity stats
- [ ] **Visualizations**:
  - [ ] Loss curves (training + validation)
  - [ ] Attention heatmaps (early & late layers)
  - [ ] Attention entropy per layer
- [ ] **Analysis**:
  - [ ] Model H vs. Model L comparison
  - [ ] Discussion of why metrics vary
  - [ ] Qualitative assessment of generation quality

### Documentation (Due: 2026-09-16)
- [ ] Final report (Phase 2 summary + figures)
- [ ] README with reproduction steps
- [ ] Google Drive links (checkpoints, datasets if needed)

---

## FAQ

**Q: Why is this deadline past?**
A: Phase 2 was supposed to finish by Sep 5. We're now generating the evaluation metrics to prepare for Phase 3 and final submission.

**Q: Can we re-grade Phase 2 later?**
A: No. Per the PDF: "No re-evaluation of closed phases." But Phase 3 report consolidates Phase 1-2 figures.

**Q: Do we need to retrain?**
A: No. Checkpoints are good. We're just evaluating and reporting them.

**Q: What about Phase 3?**
A: Starts after Phase 2 is documented. Involves reasoning finetuning (30 sep 2026).

---

## Command Quick Reference

```bash
# Run full evaluation (10K samples per model)
python eval_phase2_metrics.py --language both

# Evaluate just Telugu
python eval_phase2_metrics.py --language telugu

# Check if you have required dependencies
python -c "import torch; import tokenizers; import sacrebleu; print('✅ All dependencies OK')"

# View training logs
tail -50 telugu/model/outputs/checkpoints/training_telugu.log
tail -50 bhojpuri/model/outputs/checkpoints/training_bhojpuri.log

# Check checkpoint sizes
ls -lh telugu/model/outputs/checkpoints/checkpoint_best.pt
ls -lh bhojpuri/model/outputs/checkpoints/checkpoint_best.pt
```

---

**Last Updated**: 2026-09-07  
**Next Review**: After evaluation results are generated
