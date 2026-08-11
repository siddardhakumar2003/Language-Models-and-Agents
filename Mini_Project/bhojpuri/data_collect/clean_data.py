#!/usr/bin/env python3
"""
Clean existing Bhojpuri scraped data (from data/processed/ or data/raw/)
and save cleaned JSONL to data/cleaned/

Usage:
    python3 clean_data.py

This script:
1. Finds all JSONL files in data/processed/
2. Applies cleaning and validation
3. Removes duplicates
4. Saves cleaned files to data/cleaned/
5. Creates a merge to data/bhoj.txt
"""

import sys
from pathlib import Path

# Add collect_data to path so we can import the cleaner
sys.path.insert(0, str(Path(__file__).parent))

from collect_data.data_cleaner import BhojpuriDataCleaner

def main():
    """Run cleaning pipeline on existing data"""

    script_dir = Path(__file__).resolve().parent

    # Input: data/processed/ (where scraped data is)
    input_dir = script_dir / "data" / "processed"

    # Output: data/cleaned/ (where cleaned data goes)
    output_dir = script_dir / "data" / "cleaned"

    # Also create bhoj.txt for final output
    output_file = script_dir / "data" / "bhoj.txt"

    print("=" * 80)
    print("BHOJPURI DATA CLEANING PIPELINE")
    print("=" * 80)
    print(f"Input directory (scraped data): {input_dir}")
    print(f"Output directory (cleaned data): {output_dir}")
    print(f"Final output file: {output_file}")
    print("=" * 80)

    # Check if input directory exists and has data
    if not input_dir.exists():
        print(f"\n✗ Error: Input directory not found: {input_dir}")
        print(f"\nNo scraped data found. Make sure you have run:")
        print(f"  python3 run_collection.py --mode scrape")
        return False

    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"\n✗ No JSONL files found in {input_dir}")
        print(f"\nMake sure you have run the scraper first:")
        print(f"  python3 run_collection.py --mode scrape")
        return False

    print(f"\n✓ Found {len(jsonl_files)} files to clean")
    print("\nRunning data cleaner...")
    print("-" * 80)

    # Run cleaner
    try:
        cleaner = BhojpuriDataCleaner(
            input_dir=str(input_dir),
            output_dir=str(output_dir)
        )
        cleaner.run_cleaning(batch_size=1000)

        print("\n" + "=" * 80)
        print("✓ CLEANING COMPLETE")
        print("=" * 80)

        # Now merge to single file
        print("\nMerging cleaned data to single file...")
        print("-" * 80)

        merge_cleaned_to_txt(output_dir, output_file)

        return True

    except Exception as e:
        print(f"\n✗ Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_cleaned_to_txt(cleaned_dir, output_file):
    """Merge all cleaned JSONL files to single txt file"""

    import json

    cleaned_files = sorted(cleaned_dir.glob("cleaned_*.jsonl"))

    if not cleaned_files:
        print(f"⚠ No cleaned files found in {cleaned_dir}")
        return

    print(f"Found {len(cleaned_files)} cleaned files")

    total_lines = 0
    total_tokens = 0
    error_count = 0

    with open(output_file, 'w', encoding='utf-8') as outf:
        for idx, jsonl_file in enumerate(cleaned_files, 1):
            file_lines = 0
            file_tokens = 0

            try:
                with open(jsonl_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        try:
                            data = json.loads(line.strip())
                            text = data.get('text', '').strip()
                            tokens = data.get('tokens', 0)

                            if text:
                                outf.write(text + '\n')
                                file_lines += 1
                                file_tokens += tokens
                                total_lines += 1
                                total_tokens += tokens

                        except json.JSONDecodeError:
                            error_count += 1

                status = f"[{idx}/{len(cleaned_files)}] {jsonl_file.name}"
                tokens_str = f"{file_tokens/1e6:.2f}M" if file_tokens > 0 else "?"
                print(f"  ✓ {status:<45} {file_lines:>6} texts  {tokens_str}M tokens")

            except Exception as e:
                print(f"  ✗ Error processing {jsonl_file.name}: {e}")
                error_count += 1

    # Show final stats
    print("-" * 80)

    if output_file.exists():
        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        if total_tokens == 0:
            total_tokens = output_file.stat().st_size // 4

        print(f"\n✓ MERGE COMPLETE!")
        print(f"\nOutput file: {output_file}")
        print(f"Total lines: {total_lines:,}")
        print(f"Total tokens: {total_tokens/1e6:,.2f}M")
        print(f"File size: {file_size_mb:,.2f} MB")

        if error_count > 0:
            print(f"Errors: {error_count}")

        print("\n" + "=" * 80)
        print("✓ Data ready for tokenization!")
        print("=" * 80)
        print("\nNext step: Train tokenizer")
        print("  cd tokenizer && python3 train_tokenizer.py")
        print("=" * 80 + "\n")

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
