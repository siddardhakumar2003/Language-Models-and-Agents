#!/usr/bin/env python3
"""Quick test of Phase 1b pipeline components."""

import sys
sys.path.insert(0, "/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project")

from telugu.data_collect.archive_org_downloader import TeluguArchiveOrgDownloader
from telugu.data_collect.news_crawler import TeluguNewsCrawler
from telugu.data_collect.data_cleaner import TeluguDataCleaner
from telugu.data_collect.ocr_merge import merge_ocr_into_splits
from pathlib import Path
import time

data_dir = Path("/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project/telugu/data")

print("=" * 70)
print("TESTING PHASE 1b PIPELINE")
print("=" * 70)

# Test 1: Archive.org downloader
print("\n[1/4] Testing archive.org downloader...")
try:
    downloader = TeluguArchiveOrgDownloader(output_dir=str(data_dir / "ocr_raw"))
    result = downloader.run(target_new_items=2)
    print(f"  ✓ Downloaded {result['items_downloaded']} books")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: News crawler
print("\n[2/4] Testing news crawler...")
try:
    crawler = TeluguNewsCrawler(output_dir=str(data_dir / "ocr_raw"))
    result = crawler.run(target_new_articles=5)
    print(f"  ✓ Crawled {result['articles_fetched']} articles")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Data cleaner
print("\n[3/4] Testing data cleaner...")
try:
    cleaner = TeluguDataCleaner(
        input_dir=str(data_dir / "ocr_raw"),
        output_dir=str(data_dir / "ocr_cleaned")
    )
    cleaner.run_cleaning()
    print(f"  ✓ Cleaning completed")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Merge
print("\n[4/4] Testing merge...")
try:
    result = merge_ocr_into_splits(data_dir)
    print(f"  ✓ Merged {result.get('total_lines', 0)} lines")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

# Check results
import os
if (data_dir / "telugu.txt").exists():
    size = (data_dir / "telugu.txt").stat().st_size
    print(f"\n✓ telugu.txt exists: {size / 1024:.1f} KB")

log_file = data_dir / "phase1b_progress.log"
if log_file.exists():
    print(f"✓ Progress log exists")
    print("\nLast 5 log lines:")
    with open(log_file, 'r') as f:
        lines = f.readlines()[-5:]
        for line in lines:
            print(f"  {line.rstrip()}")
else:
    print("✗ Progress log not found")

