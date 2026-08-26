#!/usr/bin/env python3
"""
Prepare Kaggle dataset bundle for Bhojpuri pretraining.
Creates a self-contained folder that mirrors the bhojpuri/ structure:
  kaggle_bundle/
  ├── data/           (train.bin, val.bin, test.bin)
  ├── configs/        (model_config.json, tokenizer_config.json, training_config.json)
  ├── model/          (transformer.py, __init__.py)
  ├── train/          (train.py, dataset.py)
  ├── tokenizer/      (tokenizer_wrapper.py + full_wordPiece_level/bhojpuri_wp_tokenizer.json)
  └── dataset-metadata.json

The structure mirrors bhojpuri/ so that when uploaded to Kaggle and mounted at
/kaggle/input/lma-bhojpuri-phase2, sys.path.insert(0, ROOT_DIR) makes imports work unchanged.
"""

import json
import shutil
from pathlib import Path
from prepare_bin import prepare_bin


def create_kaggle_bundle(output_dir="kaggle_bundle"):
    """
    Create self-contained Kaggle dataset folder with nested language-mirror structure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    script_dir = Path(__file__).parent.parent

    print(f"Preparing Kaggle bundle (mirrored structure) in {output_path}...")

    # 1. Prepare binary corpora
    print("\n1. Preparing binary corpora...")
    for split in ["train", "val", "test"]:
        bin_path = script_dir / "data" / f"{split}.bin"
        if bin_path.exists():
            print(f"  {split}: already prepared")
        else:
            print(f"  Preparing {split}...")
            prepare_bin(split, sample_lines=None, batch_size=1000)

    # 2. Create subdirectories
    print("\n2. Creating subdirectory structure...")
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "configs").mkdir(exist_ok=True)
    (output_path / "model").mkdir(exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "tokenizer").mkdir(exist_ok=True)

    # 3. Copy .bin files and metadata
    print("\n3. Copying tokenized data...")
    for split in ["train", "val", "test"]:
        src = script_dir / "data" / f"{split}.bin"
        dst = output_path / "data" / f"{split}.bin"
        if src.exists():
            shutil.copy2(src, dst)
            size_gb = dst.stat().st_size / 1e9
            print(f"  Copied {split}.bin ({size_gb:.2f} GB)")

        # Copy metadata if exists
        meta_src = script_dir / "data" / f"{split}_metadata.json"
        if meta_src.exists():
            shutil.copy2(meta_src, output_path / "data" / f"{split}_metadata.json")

    # 4. Copy configs
    print("\n4. Copying configs...")
    for fname in ["model_config.json", "tokenizer_config.json", "training_config.json"]:
        src = script_dir / "configs" / fname
        dst = output_path / "configs" / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # 5. Copy model files
    print("\n5. Copying model files...")
    model_src = script_dir / "model"
    model_dst = output_path / "model"
    for f in model_src.glob("*.py"):
        shutil.copy2(f, model_dst / f.name)
        print(f"  Copied {f.name}")

    # 6. Copy train files (excluding notebooks, __pycache__)
    print("\n6. Copying train files...")
    train_src = script_dir / "train"
    train_dst = output_path / "train"
    for f in train_src.glob("*.py"):
        shutil.copy2(f, train_dst / f.name)
        print(f"  Copied {f.name}")

    # 7. Copy tokenizer wrapper
    print("\n7. Copying tokenizer files...")
    tokenizer_src = script_dir / "tokenizer"
    tokenizer_dst = output_path / "tokenizer"

    # Copy wrapper
    wrapper_src = tokenizer_src / "tokenizer_wrapper.py"
    if wrapper_src.exists():
        shutil.copy2(wrapper_src, tokenizer_dst / "tokenizer_wrapper.py")
        print(f"  Copied tokenizer_wrapper.py")

    # Copy tokenizer JSON to tokenizer/full_wordPiece_level/
    (tokenizer_dst / "full_wordPiece_level").mkdir(exist_ok=True)
    tokenizer_json_src = tokenizer_src / "full_wordPiece_level" / "bhojpuri_wp_tokenizer.json"
    if tokenizer_json_src.exists():
        shutil.copy2(
            tokenizer_json_src,
            tokenizer_dst / "full_wordPiece_level" / "bhojpuri_wp_tokenizer.json"
        )
        print(f"  Copied bhojpuri_wp_tokenizer.json")

    # 8. Create dataset-metadata.json
    print("\n8. Creating dataset-metadata.json...")
    metadata = {
        "id": "lma-bhojpuri-phase2",
        "title": "LMA Phase 2: Bhojpuri Transformer Pretraining",
        "subtitle": "Code + Pre-tokenized Data Bundle",
        "description": "Self-contained bundle: code (model/, train/), data (data/*.bin), configs, and tokenizer. Ready for Kaggle training.",
        "resources": [
            {"path": "data/train.bin", "description": "Pre-tokenized training corpus (binary)"},
            {"path": "data/val.bin", "description": "Pre-tokenized validation corpus (binary)"},
            {"path": "data/test.bin", "description": "Pre-tokenized test corpus (binary)"},
            {"path": "configs/model_config.json", "description": "Model architecture config"},
            {"path": "configs/tokenizer_config.json", "description": "Tokenizer config"},
            {"path": "configs/training_config.json", "description": "Training hyperparameters"},
            {"path": "model/transformer.py", "description": "BhojpuriTransformer implementation"},
            {"path": "train/train.py", "description": "Training script (Trainer class + main)"},
            {"path": "train/dataset.py", "description": "PackedLMDataset class"},
            {"path": "tokenizer/tokenizer_wrapper.py", "description": "BhojpuriTokenizer wrapper"},
            {"path": "tokenizer/full_wordPiece_level/bhojpuri_wp_tokenizer.json", "description": "WordPiece tokenizer"},
        ],
        "licenseName": "CC0",
        "isPrivate": True,
    }

    with open(output_path / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Created dataset-metadata.json")

    print(f"\n✅ Kaggle bundle ready at {output_path}/")
    print(f"\nNext steps:")
    print(f"  1. Upload to Kaggle:")
    print(f"     python3 upload_to_kaggle.py bhojpuri {output_dir}/")
    print(f"  2. Create Kaggle notebook and mount dataset as input")
    print(f"  3. Run pretrain_kaggle.ipynb")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="kaggle_bundle")
    args = parser.parse_args()
    create_kaggle_bundle(args.output_dir)
