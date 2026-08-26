#!/usr/bin/env python3
"""
Upload prepared LMA Phase 2 datasets to Kaggle.
Works for both Telugu and Bhojpuri.

Usage:
    python3 upload_to_kaggle.py telugu telugu/train/kaggle_dataset/
    python3 upload_to_kaggle.py bhojpuri bhojpuri/train/kaggle_dataset/
"""

import json
import subprocess
import sys
from pathlib import Path


def upload_to_kaggle(language: str, dataset_path: str):
    """Upload dataset bundle to Kaggle."""
    dataset_path = Path(dataset_path).resolve()
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    dataset_slug = f"lma-{language}-phase2"
    
    print(f"\n📤 Uploading {language.upper()} dataset to Kaggle...")
    print(f"   Path: {dataset_path}")
    print(f"   ID: {dataset_slug}")
    
    # Ensure metadata exists
    metadata_file = dataset_path / "dataset-metadata.json"
    if not metadata_file.exists():
        metadata = {
            "id": dataset_slug,
            "title": f"LMA Phase 2: {language.upper()} Language Model Pre-training",
            "licenses": [{"name": "CC0-1.0"}],
            "resources": [
                {"path": f.name, "description": f.name} 
                for f in dataset_path.glob("*") if f.is_file()
            ]
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Upload
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(dataset_path), "--private"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Upload successful!")
            print(f"   Access: https://kaggle.com/datasets/<username>/{dataset_slug}")
            return True
        else:
            print(f"❌ Upload failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"❌ Kaggle CLI not found. Install: pip install kaggle")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <language> <dataset_path>")
        print(f"  python3 {sys.argv[0]} telugu telugu/train/kaggle_dataset/")
        sys.exit(1)
    
    language = sys.argv[1]
    dataset_path = sys.argv[2]
    
    if language not in ["telugu", "bhojpuri"]:
        print(f"❌ Language must be 'telugu' or 'bhojpuri'")
        sys.exit(1)
    
    success = upload_to_kaggle(language, dataset_path)
    sys.exit(0 if success else 1)
