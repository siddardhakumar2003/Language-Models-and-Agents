#!/usr/bin/env python3
"""
Standalone script to extract cleaned JSONL files to bhoj.txt

Usage:
    python3 extract_to_txt.py

This script:
1. Looks for cleaned JSONL files in data/cleaned/
2. Extracts text from each file
3. Saves line-by-line to data/bhoj.txt
4. Shows statistics (tokens, file size, etc.)
"""

import json
import sys
from pathlib import Path

def extract_cleaned_to_txt():
    """Extract all cleaned JSONL files to single txt file"""

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    cleaned_dir = script_dir / "data" / "cleaned"
    output_file = script_dir / "data" / "bhoj.txt"

    print("=" * 80)
    print("BHOJPURI CLEANED DATA EXTRACTION")
    print("=" * 80)
    print(f"Input directory: {cleaned_dir}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    # Check input directory
    if not cleaned_dir.exists():
        print(f"\n✗ Error: Cleaned data directory not found!")
        print(f"  Expected: {cleaned_dir}")
        print(f"\nMake sure you have run the cleaning stage:")
        print(f"  python3 run_collection.py --mode clean")
        return False

    # Find cleaned files
    jsonl_files = sorted(cleaned_dir.glob("cleaned_*.jsonl"))

    if not jsonl_files:
        print(f"\n✗ No cleaned JSONL files found in {cleaned_dir}")
        print(f"\nCheck data/cleaned/ directory for cleaned_*.jsonl files")
        return False

    print(f"\n✓ Found {len(jsonl_files)} cleaned JSONL files\n")

    # Extract
    total_texts = 0
    total_tokens = 0
    error_count = 0

    print("Extracting text...")
    print("-" * 80)

    with open(output_file, 'w', encoding='utf-8') as outf:
        for idx, jsonl_file in enumerate(jsonl_files, 1):
            file_texts = 0
            file_tokens = 0

            try:
                with open(jsonl_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        try:
                            data = json.loads(line.strip())
                            text = data.get("text", "").strip()
                            tokens = data.get("tokens", 0)

                            if text:
                                outf.write(text + '\n')
                                file_texts += 1
                                file_tokens += tokens
                                total_texts += 1
                                total_tokens += tokens

                        except json.JSONDecodeError:
                            error_count += 1

                # Show progress
                status = f"[{idx}/{len(jsonl_files)}] {jsonl_file.name}"
                tokens_str = f"{file_tokens/1e6:.2f}M tokens" if file_tokens > 0 else "tokens unknown"
                print(f"  ✓ {status:<40} {file_texts:>6} texts  {tokens_str}")

            except Exception as e:
                print(f"  ✗ {jsonl_file.name}: {e}")
                error_count += 1

    # Show results
    print("-" * 80)

    if output_file.exists():
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        file_size_gb = output_file.stat().st_size / (1024 * 1024 * 1024)

        # Estimate tokens if not counted
        if total_tokens == 0:
            total_tokens = output_file.stat().st_size // 4

        print(f"\n✓ EXTRACTION SUCCESSFUL!")
        print(f"\nOutput file: {output_file}")
        print(f"Total texts extracted: {total_texts:,}")
        print(f"Total tokens: {total_tokens/1e6:,.2f}M")
        print(f"File size: {file_size_mb:,.2f} MB ({file_size_gb:,.2f} GB)")

        if error_count > 0:
            print(f"\n⚠ Errors encountered: {error_count}")

        print("\n" + "=" * 80)
        print("✓ Ready for tokenization and model training!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Train tokenizer: python3 tokenizer/train_tokenizer.py")
        print("  2. Train model: python3 model/train.py")
        print("=" * 80 + "\n")

        return True
    else:
        print(f"\n✗ Failed to create output file: {output_file}")
        return False

if __name__ == '__main__':
    success = extract_cleaned_to_txt()
    sys.exit(0 if success else 1)
