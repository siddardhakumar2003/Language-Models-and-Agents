#!/usr/bin/env python3
"""
Clean Telugu dataset (memory-efficient line-by-line processing):
1. Remove ASCII periods "."
2. Replace multiple consecutive dashes (-+) with single dash (-)

Usage:
    python3 clean_telugu.py
"""

import re
from pathlib import Path
import tempfile
import shutil

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SPLITS = ["train", "val", "test"]
FILENAME = "te.txt"

def clean_file_lines(file_path):
    """Process file line-by-line to save memory."""
    print(f"  {file_path.name}...", end=" ", flush=True)

    periods_removed = 0
    dashes_removed = 0

    # Create temp file
    temp_fd, temp_path = tempfile.mkstemp()

    try:
        with open(temp_path, 'w', encoding='utf-8') as temp_file:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Count before
                    periods_before = line.count(".")
                    dashes_before = line.count("-")

                    # Remove periods
                    cleaned = line.replace(".", "")

                    # Replace multiple dashes
                    cleaned = re.sub(r"-+", "-", cleaned)

                    # Count after
                    periods_after = cleaned.count(".")
                    dashes_after = cleaned.count("-")

                    periods_removed += periods_before - periods_after
                    dashes_removed += dashes_before - dashes_after

                    # Write to temp
                    temp_file.write(cleaned)

        # Replace original with cleaned
        shutil.move(temp_path, file_path)
        print(f"✓ (Removed {periods_removed} periods, {dashes_removed} extra dashes)")
        return periods_removed, dashes_removed

    except Exception as e:
        print(f"✗ Error: {e}")
        return 0, 0
    finally:
        # Cleanup temp file if still exists
        if Path(temp_path).exists():
            Path(temp_path).unlink()

def main():
    print("=" * 70)
    print("CLEANING TELUGU: Remove Periods + Clean Dashes (Memory-Efficient)")
    print("=" * 70)

    total_periods_removed = 0
    total_dashes_removed = 0

    for split in SPLITS:
        file_path = DATA_ROOT / split / FILENAME

        if not file_path.exists():
            print(f"⚠️  {split}/{FILENAME} not found, skipping...")
            continue

        periods, dashes = clean_file_lines(file_path)
        total_periods_removed += periods
        total_dashes_removed += dashes

    print("=" * 70)
    print(f"✅ Telugu cleaning complete!")
    print(f"   Total periods removed: {total_periods_removed}")
    print(f"   Total extra dashes removed: {total_dashes_removed}")
    print("=" * 70)

if __name__ == "__main__":
    main()
