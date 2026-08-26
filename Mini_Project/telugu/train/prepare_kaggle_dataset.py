#!/usr/bin/env python3
"""
Prepare Kaggle dataset bundle for Telugu pretraining.
Run locally: creates kaggle_dataset/ folder with .bin + tokenizer + configs ready to upload.
"""

import json
import shutil
from pathlib import Path
from prepare_bin import prepare_bin


def create_kaggle_bundle(output_dir="kaggle_dataset"):
    """
    Create self-contained Kaggle dataset folder with:
    - train.bin, val.bin, test.bin (pre-tokenized)
    - tokenizer JSON
    - model_config.json, tokenizer_config.json
    - dataset-metadata.json (Kaggle format)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    script_dir = Path(__file__).parent.parent

    print(f"Preparing Kaggle bundle in {output_path}...")

    print("\n1. Preparing binary corpora...")
    for split in ["train", "val", "test"]:
        bin_path = script_dir / "data" / split / f"{split}.bin"
        if bin_path.exists():
            print(f"  {split}: already prepared")
        else:
            print(f"  Preparing {split}...")
            prepare_bin(split, sample_lines=None, batch_size=1000)

    print("\n2. Copying .bin files...")
    for split in ["train", "val", "test"]:
        src = script_dir / "data" / split / f"{split}.bin"
        dst = output_path / f"{split}.bin"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {split}.bin ({dst.stat().st_size / 1e9:.2f} GB)")

    print("\n3. Copying tokenizer...")
    tokenizer_src = script_dir / "tokenizer" / "full_wordPiece_level" / "telugu_wp_tokenizer.json"
    tokenizer_dst = output_path / "telugu_wp_tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy2(tokenizer_src, tokenizer_dst)
        print(f"  Copied tokenizer")

    print("\n4. Copying configs...")
    for fname in ["model_config.json", "tokenizer_config.json", "training_config.json"]:
        src = script_dir / "configs" / fname
        dst = output_path / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    print("\n5. Creating dataset-metadata.json...")
    metadata = {
        "id": "telugu-lm-pretrain-phase2",
        "title": "Telugu Transformer LM - Phase 2 Pretraining Data",
        "subtitle": "Pre-tokenized binary corpus + configs for Telugu decoder-only Transformer",
        "description": "Complete dataset bundle for Phase 2 pretraining: train/val/test .bin files, tokenizer, and model configs. Ready for Kaggle training.",
        "resources": [
            {"path": f"{split}.bin", "description": f"{split} corpus (binary)"} for split in ["train", "val", "test"]
        ] + [
            {"path": "telugu_wp_tokenizer.json", "description": "WordPiece tokenizer"},
            {"path": "model_config.json", "description": "Model architecture config"},
            {"path": "tokenizer_config.json", "description": "Tokenizer config"},
            {"path": "training_config.json", "description": "Training hyperparameters"},
        ],
        "licenseName": "CC0",
        "isPrivate": False,
    }

    with open(output_path / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Created dataset-metadata.json")

    print(f"\n✓ Kaggle dataset bundle ready at {output_path}/")
    print(f"\nNext step: Upload to Kaggle")
    print(f"  Option 1: Web UI - go to kaggle.com/datasets/upload/new and drag-drop {output_path}")
    print(f"  Option 2: CLI - kaggle datasets create -p {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="kaggle_dataset")
    args = parser.parse_args()
    create_kaggle_bundle(args.output_dir)
