#!/usr/bin/env python3
"""
Clean existing Telugu scraped data and extract to telugu.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_cleaner import TeluguDataCleaner

def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "cleaned"
    output_file = script_dir / "data" / "telugu.txt"

    print("=" * 80)
    print("TELUGU DATA CLEANING & EXTRACTION")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    if not input_dir.exists():
        print(f"\n✗ Error: Cleaned data directory not found: {input_dir}")
        print(f"Make sure you have run the cleaning stage first:")
        print(f"  python3 -m collect_data.pipeline --mode clean")
        return False

    # Extract cleaned files
    import json
    
    jsonl_files = sorted(input_dir.glob("cleaned_*.jsonl"))
    
    if not jsonl_files:
        print(f"\n✗ No cleaned JSONL files found in {input_dir}")
        return False

    print(f"\n✓ Found {len(jsonl_files)} cleaned JSONL files\n")
    
    total_texts = 0
    total_tokens = 0
    
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
                            pass
                
                status = f"[{idx}/{len(jsonl_files)}] {jsonl_file.name}"
                tokens_str = f"{file_tokens/1e6:.2f}M tokens" if file_tokens > 0 else "tokens unknown"
                print(f"  ✓ {status:<40} {file_texts:>6} texts  {tokens_str}")
            
            except Exception as e:
                print(f"  ✗ {jsonl_file.name}: {e}")
    
    print("-" * 80)
    
    if output_file.exists():
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        file_size_gb = output_file.stat().st_size / (1024 * 1024 * 1024)
        
        if total_tokens == 0:
            total_tokens = output_file.stat().st_size // 4
        
        print(f"\n✓ EXTRACTION SUCCESSFUL!")
        print(f"\nOutput file: {output_file}")
        print(f"Total texts: {total_texts:,}")
        print(f"Total tokens: {total_tokens/1e6:,.2f}M")
        print(f"File size: {file_size_mb:,.2f} MB ({file_size_gb:,.2f} GB)")
        print("\n" + "=" * 80)
        return True
    
    return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
