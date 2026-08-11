#!/usr/bin/env python3
"""
Extract text from cleaned JSONL files and save line by line to text file.

Usage from telugu/: python3 collect_data/extract_text.py
"""

import json
import sys
from pathlib import Path

# Determine telugu directory (script location aware)
script_dir = Path(__file__).resolve().parent
telugu_dir = script_dir.parent
cleaned_data_dir = telugu_dir / "data" / "cleaned"
output_file = telugu_dir / "data" / "telugu.txt"

if not cleaned_data_dir.exists():
    print(f"Error: Cleaned data directory not found: {cleaned_data_dir}")
    exit(1)

# Find all JSONL files in cleaned folder
jsonl_files = sorted(cleaned_data_dir.glob("*.jsonl"))
print(f"Found {len(jsonl_files)} JSONL files")

if not jsonl_files:
    print("No JSONL files found!")
    exit(1)

print(f"Output file: {output_file}")
print("=" * 60)

# Extract text from all JSONL files
total_lines = 0
error_count = 0

with open(output_file, 'w', encoding='utf-8') as outf:
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.name}")

        with open(jsonl_file, 'r', encoding='utf-8') as inf:
            file_lines = 0
            for line_num, line in enumerate(inf, 1):
                try:
                    # Parse JSON
                    data = json.loads(line.strip())

                    # Extract text field
                    text = data.get("text", "").strip()

                    if text:  # Only write non-empty text
                        outf.write(text + '\n')
                        file_lines += 1
                        total_lines += 1
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"  ⚠ Error at line {line_num}: {e}")
                except Exception as e:
                    error_count += 1
                    print(f"  ⚠ Unexpected error at line {line_num}: {e}")

        print(f"  ✓ Extracted {file_lines} texts")

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"✓ Total texts extracted: {total_lines}")
print(f"⚠ Errors encountered: {error_count}")
print(f"✓ Output file: {output_file}")
print(f"✓ File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
print("=" * 60)
