#!/usr/bin/env python3
"""
Extract text from cleaned JSONL files and save line by line to bhoj.txt

Usage:
    python3 extract_text.py

Or from bhojpuri root:
    python3 collect_data/extract_text.py
"""

import json
import sys
from pathlib import Path

# Determine bhojpuri directory (script location aware)
script_dir = Path(__file__).resolve().parent
bhojpuri_dir = script_dir.parent
cleaned_data_dir = bhojpuri_dir / "data" / "cleaned"
output_file = bhojpuri_dir / "data" / "bhoj.txt"

print("=" * 70)
print("BHOJPURI TEXT EXTRACTION")
print("=" * 70)
print(f"Input directory: {cleaned_data_dir}")
print(f"Output file: {output_file}")
print("=" * 70)

# Check if cleaned directory exists
if not cleaned_data_dir.exists():
    print(f"\n✗ Error: Cleaned data directory not found: {cleaned_data_dir}")
    print(f"\nMake sure you have run the cleaning stage first:")
    print(f"  python3 run_collection.py --mode clean")
    sys.exit(1)

# Find all JSONL files in cleaned folder
jsonl_files = sorted(cleaned_data_dir.glob("cleaned_*.jsonl"))
print(f"\n✓ Found {len(jsonl_files)} cleaned JSONL files")

if not jsonl_files:
    print(f"\n✗ No JSONL files found in {cleaned_data_dir}")
    print(f"Check if data cleaning has completed.")
    sys.exit(1)

# Extract text from all JSONL files
total_lines = 0
error_count = 0
total_tokens = 0

print(f"\nExtracting text from cleaned files...")
print("-" * 70)

with open(output_file, 'w', encoding='utf-8') as outf:
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.name}")

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as inf:
                file_lines = 0
                file_tokens = 0

                for line_num, line in enumerate(inf, 1):
                    try:
                        # Parse JSON
                        data = json.loads(line.strip())

                        # Extract text field
                        text = data.get("text", "").strip()
                        tokens = data.get("tokens", 0)

                        if text:  # Only write non-empty text
                            outf.write(text + '\n')
                            file_lines += 1
                            total_lines += 1
                            file_tokens += tokens
                            total_tokens += tokens

                    except json.JSONDecodeError as e:
                        error_count += 1
                        if line_num <= 3:  # Only show first few errors
                            print(f"  ⚠ JSON error at line {line_num}: {e}")
                    except Exception as e:
                        error_count += 1
                        if line_num <= 3:
                            print(f"  ⚠ Error at line {line_num}: {e}")

                print(f"  ✓ Extracted {file_lines} texts from {jsonl_file.name}")
                print(f"    (Tokens: {file_tokens/1e6:.2f}M)")

        except Exception as e:
            print(f"  ✗ Failed to process {jsonl_file.name}: {e}")
            error_count += 1

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

# Get file stats
if output_file.exists():
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    file_size_gb = output_file.stat().st_size / (1024 * 1024 * 1024)

    # Calculate estimated tokens if not available
    if total_tokens == 0:
        total_tokens = output_file.stat().st_size // 4

    print(f"\n✓ Output file: {output_file}")
    print(f"✓ Total texts extracted: {total_lines}")
    print(f"✓ Total tokens: {total_tokens/1e6:.2f}M")
    print(f"✓ File size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")

    if error_count > 0:
        print(f"\n⚠ Errors encountered: {error_count}")

    print("\n" + "=" * 70)
    print("SUCCESS - Ready for tokenization and model training!")
    print("=" * 70 + "\n")
else:
    print(f"\n✗ Failed to create output file: {output_file}")
    sys.exit(1)
