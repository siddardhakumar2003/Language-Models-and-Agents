#!/usr/bin/env python3
"""
Upload all 6 trained tokenizers to Kaggle properly
Organized structure with only necessary files
"""

import os
import shutil
import json
from pathlib import Path

# Kaggle dataset handle
KAGGLE_HANDLE = "kspsvlnsiddardha/lma-tokenizers"

# Base paths
BASE_DIR = Path("/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project")
BUFFER_DIR = Path("/tmp/kaggle-tokenizers")

# Clean up previous buffer if exists
if BUFFER_DIR.exists():
    shutil.rmtree(BUFFER_DIR)
BUFFER_DIR.mkdir(parents=True)

print("="*70)
print("PREPARING TOKENIZERS FOR KAGGLE UPLOAD")
print("="*70)
print("")

# Define tokenizer sources and destinations
tokenizers_config = {
    "telugu": {
        "byte_level": {
            "source_dir": BASE_DIR / "telugu/tokenizer/full_byte_level",
            "files": ["telugu_tokenizer_full.json", "tokenizer_config_full.json", "telugu_tokenizer_report_full.json"],
            "vocab_size": 50000
        },
        "unicode_level": {
            "source_dir": BASE_DIR / "telugu/tokenizer/full_unicode_level",
            "files": ["checkpoint_batch_140.json", "tokenizer_config_full.json"],
            "vocab_size": 50000
        },
        "wordpiece": {
            "source_dir": BASE_DIR / "telugu/tokenizer/full_wordPiece_level",
            "files": ["telugu_wp_tokenizer.json", "wp_tokenizer_config.json"],
            "vocab_size": 50000
        }
    },
    "bhojpuri": {
        "byte_level": {
            "source_dir": BASE_DIR / "bhojpuri/tokenizer/full_byte_level",
            "files": ["bhoj_tokenizer_full.json", "tokenizer_config_full.json", "bhoj_tokenizer_report_full.json"],
            "vocab_size": 8000
        },
        "unicode_level": {
            "source_dir": BASE_DIR / "bhojpuri/tokenizer/full_unicode_level",
            "files": ["bhojpuri_tokenizer_full.json", "tokenizer_config_full.json", "bhoj_tokenizer_report_full.json"],
            "vocab_size": 32000
        },
        "wordpiece": {
            "source_dir": BASE_DIR / "bhojpuri/tokenizer/full_wordPiece_level",
            "files": ["bhojpuri_wp_tokenizer.json", "wp_tokenizer_config.json"],
            "vocab_size": 32000
        }
    }
}

# Copy tokenizers to buffer directory
total_size = 0
files_copied = 0

for lang, variants in tokenizers_config.items():
    lang_dir = BUFFER_DIR / lang
    lang_dir.mkdir(exist_ok=True)

    print(f"\n{lang.upper()} Tokenizers:")
    print("─" * 70)

    for variant, config in variants.items():
        variant_dir = lang_dir / variant
        variant_dir.mkdir(exist_ok=True)

        source_dir = config["source_dir"]
        files = config["files"]
        vocab_size = config["vocab_size"]

        print(f"\n  {variant.upper()} ({vocab_size:,} vocab):")

        for file in files:
            src_file = source_dir / file
            if src_file.exists():
                dest_file = variant_dir / file
                shutil.copy2(src_file, dest_file)
                file_size = src_file.stat().st_size / 1024  # KB
                total_size += file_size
                files_copied += 1
                print(f"    ✓ {file} ({file_size:.1f} KB)")
            else:
                print(f"    ✗ {file} (NOT FOUND)")

# Create comprehensive metadata
metadata = {
    "project": "Language Models and Agents (LMA)",
    "phase": "Phase 1 - Tokenizer Training & Evaluation",
    "completion_date": "2026-08-21",
    "tokenizers": {
        "telugu": {
            "byte_level": {
                "vocab_size": 50000,
                "unique_tokens_used": 1978,
                "avg_chars_per_token": 1.3327,
                "unknown_rate": "0.0000%",
                "type": "BPE",
                "coverage": "99.0%"
            },
            "unicode_level": {
                "vocab_size": 50000,
                "unique_tokens_used": 11398,
                "avg_chars_per_token": 5.9283,
                "unknown_rate": "0.0000%",
                "type": "BPE",
                "coverage": "22.8%",
                "checkpoint": "batch_140"
            },
            "wordpiece": {
                "vocab_size": 50000,
                "unique_tokens_used": 9539,
                "avg_chars_per_token": 5.8353,
                "unknown_rate": "0.0000%",
                "type": "WordPiece"
            }
        },
        "bhojpuri": {
            "byte_level": {
                "vocab_size": 8000,
                "unique_tokens_used": 7761,
                "avg_chars_per_token": 1.5412,
                "unknown_rate": "0.0%",
                "type": "BPE",
                "coverage": "97.01%"
            },
            "unicode_level": {
                "vocab_size": 32000,
                "unique_tokens_used": 31258,
                "avg_chars_per_token": 4.348,
                "unknown_rate": "0.0%",
                "type": "BPE",
                "coverage": "97.68%"
            },
            "wordpiece": {
                "vocab_size": 32000,
                "test_tokens": 11391403,
                "avg_chars_per_token": 3.95,
                "unknown_rate": "0.0002%",
                "type": "WordPiece"
            }
        }
    },
    "data_stats": {
        "telugu": {
            "total_lines": 25349199,
            "estimated_tokens": 4421917647,
            "progress_percent": 884.4
        },
        "bhojpuri": {
            "total_lines": 1826713,
            "estimated_tokens": 291685379,
            "progress_percent": 58.3
        }
    },
    "usage": {
        "loading": "from tokenizers import Tokenizer; tok = Tokenizer.from_file('path/to/tokenizer.json')",
        "encoding": "encoded = tok.encode(text)",
        "decoding": "text = tok.decode(token_ids)"
    }
}

# Save metadata
metadata_file = BUFFER_DIR / "TOKENIZER_METADATA.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\n\n✓ Metadata saved: {metadata_file.name}")

# Create comprehensive README
readme_content = """# LMA Phase 1 Tokenizers

Complete set of trained tokenizers for the Language Models and Agents (LMA) project.
All 6 tokenizers trained and evaluated on held-out test sets with comprehensive metrics.

## Dataset Overview

### Telugu (Model H - Higher-resource)
- **Byte-Level BPE (50K vocab)**: Maximum compression (1.33 chars/token)
- **Unicode-Level BPE (50K vocab)**: Linguistic structure preservation (5.93 chars/token)
- **WordPiece (50K vocab)**: Subword decomposition with ## prefix (5.84 chars/token)
- **Data**: 25.3M lines, 4.4B tokens (884% of 500M target)
- **Quality**: 99% coverage, 0.0% unknown token rate

### Bhojpuri (Model L - Lower-resource, Phase 3 Expansion)
- **Byte-Level BPE (8K vocab)**: Compact for resource constraints (1.54 chars/token)
- **Unicode-Level BPE (32K vocab)**: Balanced representation (4.35 chars/token)
- **WordPiece (32K vocab)**: Subword strategy (3.95 chars/token)
- **Data**: 1.83M lines, 291.7M tokens (58.3% of 500M target)
- **Quality**: 97.0-97.7% coverage, 0.0-0.0002% unknown token rate

## Tokenizer Characteristics

### Vocabulary Independence
All 6 tokenizers maintain completely independent vocabularies:
- No sharing between languages
- No sharing between variants within same language
- Each can be used independently or in parallel

### Evaluation Metrics (on held-out test sets)

| Language | Variant | Vocab | Chars/Token | Coverage | UNK Rate |
|----------|---------|-------|------------|----------|----------|
| Telugu | Byte-level | 50K | 1.33 | 99.0% | 0.0000% |
| Telugu | Unicode-level | 50K | 5.93 | 22.8% | 0.0000% |
| Telugu | WordPiece | 50K | 5.84 | — | 0.0000% |
| Bhojpuri | Byte-level | 8K | 1.54 | 97.01% | 0.0% |
| Bhojpuri | Unicode-level | 32K | 4.35 | 97.68% | 0.0% |
| Bhojpuri | WordPiece | 32K | 3.95 | — | 0.0002% |

## File Structure

```
tokenizers/
├── telugu/
│   ├── byte_level/
│   │   ├── telugu_tokenizer_full.json          (main tokenizer)
│   │   ├── tokenizer_config_full.json          (config)
│   │   └── telugu_tokenizer_report_full.json   (evaluation metrics)
│   ├── unicode_level/
│   │   ├── checkpoint_batch_140.json           (best checkpoint)
│   │   └── tokenizer_config_full.json
│   └── wordpiece/
│       ├── telugu_wp_tokenizer.json
│       └── wp_tokenizer_config.json
│
├── bhojpuri/
│   ├── byte_level/
│   │   ├── bhoj_tokenizer_full.json
│   │   ├── tokenizer_config_full.json
│   │   └── bhoj_tokenizer_report_full.json
│   ├── unicode_level/
│   │   ├── bhojpuri_tokenizer_full.json
│   │   ├── tokenizer_config_full.json
│   │   └── bhoj_tokenizer_report_full.json
│   └── wordpiece/
│       ├── bhojpuri_wp_tokenizer.json
│       └── wp_tokenizer_config.json
│
└── TOKENIZER_METADATA.json                     (this file's metadata)
```

## Loading Tokenizers

### Using tokenizers library

```python
from tokenizers import Tokenizer

# Telugu
te_byte = Tokenizer.from_file("telugu/byte_level/telugu_tokenizer_full.json")
te_unicode = Tokenizer.from_file("telugu/unicode_level/checkpoint_batch_140.json")
te_wp = Tokenizer.from_file("telugu/wordpiece/telugu_wp_tokenizer.json")

# Bhojpuri
bho_byte = Tokenizer.from_file("bhojpuri/byte_level/bhoj_tokenizer_full.json")
bho_unicode = Tokenizer.from_file("bhojpuri/unicode_level/bhojpuri_tokenizer_full.json")
bho_wp = Tokenizer.from_file("bhojpuri/wordpiece/bhojpuri_wp_tokenizer.json")
```

### Tokenization Example

```python
# Telugu example
text_te = "నందమూరి తారక రామారావు కథా నాయకుడు"
encoding = te_unicode.encode(text_te)
print(f"Tokens: {encoding.tokens}")
print(f"Token count: {len(encoding.tokens)}")

# Bhojpuri example
text_bho = "भोजपुरी भारत के बिहार में बोली जाती है।"
encoding = bho_unicode.encode(text_bho)
print(f"Tokens: {encoding.tokens}")
print(f"Token count: {len(encoding.tokens)}")
```

### Decoding

```python
# Both languages
decoded = tokenizer.decode(token_ids)
```

## Training Details

All tokenizers trained on:
- **Telugu**: 20.3M training lines from te.txt corpus + web scraping + OCR augmentation
- **Bhojpuri**: 1.46M training lines from HuggingFace + OCR + machine translation (Phase 3)

Training methodology:
- Deterministic seed (42) for reproducibility
- Complete train/val/test splits (80/10/10)
- Evaluated on held-out test sets
- No vocabulary sharing between languages or variants

## Data Sources

### Telugu
- Manual corpus (te.txt): 21.5M lines
- Web scraping: 29.2K lines from Wikipedia
- OCR augmentation: 3.8M lines from archive.org

### Bhojpuri (Phase 3 Expansion)
- HuggingFace Corpus: 386K docs (~261K lines)
- OCR augmentation: 511.6K lines
- English→Bhojpuri Translation (fineweb-edu): 711.1K lines
- Hindi→Bhojpuri Translation: 15.9K lines
- Web scraping: 3.2K lines

## Citation

```
Language Models and Agents (LMA) - Phase 1 Tokenizers
Roll Number: 2025201061
Date: August 21, 2026
Institution: IIIT Hyderabad
```

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/kspsvlnsiddardha/lma-slm
- Phase 1 Report: See report/phase-1/phase1_report.tex
- Tokenizers Library: https://github.com/huggingface/tokenizers

## License

Project work for IIIT Hyderabad (Language Models and Agents - Individual Project)
"""

readme_file = BUFFER_DIR / "README.md"
with open(readme_file, 'w') as f:
    f.write(readme_content)
print(f"✓ README created: {readme_file.name}")

# Summary
print("\n" + "="*70)
print("TOKENIZER UPLOAD PREPARATION COMPLETE")
print("="*70)
print(f"\nBuffer directory: {BUFFER_DIR}")
print(f"Total files copied: {files_copied}")
print(f"Total size: {total_size/1024:.1f} MB")
print(f"\nFolder structure ready for Kaggle upload:")
print("  ✓ telugu/byte_level/")
print("  ✓ telugu/unicode_level/")
print("  ✓ telugu/wordpiece/")
print("  ✓ bhojpuri/byte_level/")
print("  ✓ bhojpuri/unicode_level/")
print("  ✓ bhojpuri/wordpiece/")
print("  ✓ TOKENIZER_METADATA.json")
print("  ✓ README.md")

print("\n" + "="*70)
print("Next step: Upload to Kaggle")
print("="*70)
print(f"\nCommand to upload:")
print(f"  kagglehub.dataset_upload('{KAGGLE_HANDLE}', '{BUFFER_DIR}', version_notes='Phase 1 Complete - All 6 Tokenizers')")
print("\nNote: Install kagglehub with: pip install kagglehub")
print("      Set KAGGLE_API_TOKEN environment variable first")
