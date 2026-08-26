# Phase 2: Model Implementation & Pretraining — Quick Start Guide

## Overview

Phase 2 implements a from-scratch decoder-only Transformer LM and pretrain Models H (Telugu) and L (Bhojpuri) independently to ~25M params / ~500M tokens each.

## What's Implemented

✅ **Model** (`telugu/model/transformer.py`, `bhojpuri/model/transformer.py`)
- CausalSelfAttention: manual Q,K,V + scaled dot-product + causal masking
- FeedForward: GELU MLP
- Block: pre-norm residuals
- TeluguTransformer / BhojpuriTransformer: ~24.7M / ~20.1M params
- generate() with temperature sampling
- Causal masking verification test in `__main__`

✅ **Data Pipeline**
- `prepare_bin.py`: tokenizes raw .txt → binary `.bin` + metadata
- `dataset.py`: PackedLMDataset reads .bin, yields contiguous windows

✅ **Training** (`telugu/train/train.py`, `bhojpuri/train/train.py`)
- AdamW + cosine warmup scheduler
- Checkpoint save/resume (checkpoint_last.pt + checkpoint_best.pt)
- Metric logging to `logs/training_<lang>.log` (JSONL)
- AMP support for CUDA

✅ **Evaluation** (`telugu/eval/evaluate.py`, `bhojpuri/eval/evaluate.py`)
- Language modeling: loss, PPL
- Generation: greedy + temperature sampling (0.5/1.0/1.5)
- Attention analysis: entropy, heatmaps

✅ **Kaggle Setup**
- `prepare_kaggle_dataset.py`: creates self-contained mirrored bundle
- Notebook: `pretrain_kaggle.ipynb` imports real training code

## Quick Start (Local CPU Testing)

### 1. Test Model

```bash
python3 telugu/model/transformer.py
python3 bhojpuri/model/transformer.py
# Should print: ✓ All tests passed!
```

### 2. Prepare Tiny Dataset (for testing)

```bash
cd telugu/train
python3 prepare_bin.py --split train --sample-lines 100 --output-dir /tmp/test
# Creates /tmp/test/train.bin + metadata
```

### 3. Tiny Training Run

```bash
cd telugu/train
python3 train.py \
  --batch-size 4 \
  --total-steps 50 \
  --eval-steps 10 \
  --checkpoint-steps 25
# Should print: ✓ Training complete!
```

### 4. Evaluate

```bash
cd telugu/eval
python3 evaluate.py --checkpoint ../train/checkpoints/checkpoint_best.pt --val-data /tmp/test/train.bin
```

## Full Scale (Kaggle GPU)

### 1. Prepare Dataset Locally

```bash
# For EACH language:
cd telugu/train
python3 prepare_kaggle_dataset.py --output-dir kaggle_bundle
# Creates telugu/train/kaggle_bundle/ with nested code+data structure
```

### 2. Upload to Kaggle

```bash
# Use the shared upload script:
python3 upload_to_kaggle.py telugu telugu/train/kaggle_bundle/

# For Bhojpuri:
python3 upload_to_kaggle.py bhojpuri bhojpuri/train/kaggle_bundle/
```

**Alternative (manual):**
- Go to kaggle.com/datasets/upload/new
- Drag-drop `telugu/train/kaggle_bundle/` folder
- Keep the suggested dataset ID

### 3. Run Kaggle Notebook

- Create new notebook on Kaggle
- Add the dataset as **Input** (mount at `/kaggle/input/lma-telugu-phase2` etc.)
- Upload `telugu/train/pretrain_kaggle.ipynb` (or create empty notebook and copy cells)
- Copy cell content from `pretrain_kaggle.ipynb` into notebook cells
- **Cell 1 only:** Modify `ROOT_DIR`, `OUT_DIR`, `CHECK_DIR` (leave `hp` values as `None` to use JSON defaults)
- **Run All** — training runs using real `Trainer` class with full checkpointing, logging, AMP

### 4. Download Checkpoints

Notebook saves to `/kaggle/working/checkpoints/`:
- `checkpoint_last.pt` — latest state (for resuming)
- `checkpoint_best.pt` — best validation loss
- Training logs in `/kaggle/working/logs/`

**For resuming multi-session training:**
1. Save notebook output as a Kaggle dataset (`+ New Version` or create new dataset from output)
2. Next run: set `CHECK_DIR = "/kaggle/input/<previous-output-dataset>/checkpoints"`
3. Restart notebook (automatically resumes from saved step)

### 5. Evaluate Locally

After downloading checkpoint:
```bash
cd telugu/eval
python3 evaluate.py \
  --checkpoint <downloaded_checkpoint_best.pt> \
  --val-data telugu/data/val.bin
```

## Configuration Files

### `configs/model_config.json`
- `vocab_size`: 50000 (Telugu), 32000 (Bhojpuri)
- `embedding_dim`: 256
- `num_layers`: 15
- `num_heads`: 8
- `hidden_dim`: 1024
- `max_seq_length`: 512

### `configs/training_config.json`
- `batch_size`: 32
- `learning_rate`: 1e-4
- `total_steps`: 500000
- `eval_steps`: 5000
- `checkpoint_steps`: 10000
- `device`: "auto" (cuda if available, else cpu)
- `amp`: true (mixed precision)

### `configs/tokenizer_config.json`
- `model_type`: "wordpiece"
- `vocab_size`: 50000 / 32000
- `tokenizer_path`: points to WordPiece .json

## Important Notes

- **Local training**: CPU is slow for large corpora. Use Kaggle for real runs.
- **Checkpoint resume**: Set `CHECK_DIR` in Kaggle notebook to continue from prior session.
- **Best model**: Loaded from `checkpoint_best.pt` (lowest validation loss).
- **Metric logging**: Check `logs/training_<lang>.log` (JSONL) for loss curves.
- **No shared weights**: Models H and L are completely independent (separate data, tokenizer, weights).
- **Bundle structure**: Mirrors telugu/ or bhojpuri/ so imports resolve unchanged (no code modification needed).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `.bin` file not found | Run `prepare_kaggle_dataset.py` first |
| OOM on Kaggle | Reduce `batch_size` in `training_config.json` |
| Training too slow | Check GPU is being used: `torch.cuda.is_available()` in Kaggle |
| Checkpoint corrupt | Delete and restart from earlier checkpoint or fresh |
| Imports fail in Kaggle | Confirm `sys.path.insert(0, ROOT_DIR)` and dataset is mounted correctly |

## Next Steps

1. Prepare and upload datasets using `prepare_kaggle_dataset.py` + `upload_to_kaggle.py`
2. Run training on Kaggle until target tokens reached (multi-session via resume)
3. Download checkpoints and run evaluation locally
4. Analyze attention patterns and generate samples
5. Proceed to Phase 3 (finetuning on reasoning tasks)

---

**Deadline**: 5 Sep 2026, 11:59 PM  
**Expected time to 500M tokens**: 1-2 weeks of continuous Kaggle runs (depending on throttling)
