#!/usr/bin/env python3
"""
Clean both Bhojpuri and Telugu datasets:
1. Remove ASCII periods "." (not meaningful in Devanagari/Telugu)
2. Replace multiple consecutive dashes (-+) with single dash (-)

Usage:
    python3 clean_both_languages.py
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LANGUAGES = {
    "bhojpuri": PROJECT_ROOT / "bhojpuri" / "data",
    "telugu": PROJECT_ROOT / "telugu" / "data",
}
SPLITS = ["train", "val", "test"]
FILENAME = "te.txt"  # Will use this for Telugu, bhoj.txt for Bhojpuri

def clean_file(file_path, language):
    """Remove periods and replace multiple dashes with single dash."""
    print(f"  {file_path.name}...", end=" ")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Count before
    period_count_before = content.count(".")
    dash_count_before = content.count("-")

    # Remove ASCII periods "."
    cleaned = content.replace(".", "")

    # Replace multiple dashes with single dash
    cleaned = re.sub(r"-+", "-", cleaned)

    # Count after
    period_count_after = cleaned.count(".")
    dash_count_after = cleaned.count("-")
    periods_removed = period_count_before - period_count_after
    dashes_removed = dash_count_before - dash_count_after

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✓ (Removed {periods_removed} periods, {dashes_removed} extra dashes)")
    return periods_removed, dashes_removed

def main():
    print("=" * 70)
    print("CLEANING BOTH LANGUAGES: Remove Periods + Clean Dashes")
    print("=" * 70)

    total_periods_removed = 0
    total_dashes_removed = 0

    for language, data_root in LANGUAGES.items():
        print(f"\n🔧 {language.upper()} Language")
        print("-" * 70)

        # Choose filename based on language
        filename = "bhoj.txt" if language == "bhojpuri" else "te.txt"

        for split in SPLITS:
            file_path = data_root / split / filename

            if not file_path.exists():
                print(f"  ⚠️  {split}/{filename} not found, skipping...")
                continue

            periods, dashes = clean_file(file_path, language)
            total_periods_removed += periods
            total_dashes_removed += dashes

    print("\n" + "=" * 70)
    print(f"✅ Cleaning complete!")
    print(f"   Total periods removed: {total_periods_removed}")
    print(f"   Total extra dashes removed: {total_dashes_removed}")
    print("=" * 70)

if __name__ == "__main__":
    main()
