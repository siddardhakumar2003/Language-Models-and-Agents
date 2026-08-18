#!/usr/bin/env python3
"""
Clean Bhojpuri dataset: Replace multiple consecutive dashes with single dash.

Usage:
    python3 clean_dashes.py
"""

import re
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SPLITS = ["train", "val", "test"]
FILENAME = "bhoj.txt"

def clean_dashes_in_file(file_path):
    """Replace -+ (one or more dashes) with single dash."""
    print(f"Processing: {file_path.name}...", end=" ")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Count occurrences before
    dash_count_before = content.count("-")

    # Replace multiple dashes with single dash
    # -+ means one or more dash characters
    cleaned = re.sub(r"-+", "-", content)

    # Count occurrences after
    dash_count_after = cleaned.count("-")
    dashes_removed = dash_count_before - dash_count_after

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✓ (Removed {dashes_removed} extra dashes: {dash_count_before} → {dash_count_after})")
    return dashes_removed

def main():
    print("=" * 70)
    print("CLEANING BHOJPURI DATASET: Multiple Dashes → Single Dash")
    print("=" * 70)

    total_removed = 0

    for split in SPLITS:
        file_path = DATA_ROOT / split / FILENAME

        if not file_path.exists():
            print(f"⚠️  {split}/{FILENAME} not found, skipping...")
            continue

        removed = clean_dashes_in_file(file_path)
        total_removed += removed

    print("=" * 70)
    print(f"✅ Cleaning complete! Total extra dashes removed: {total_removed}")
    print("=" * 70)

if __name__ == "__main__":
    main()
