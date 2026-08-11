#!/usr/bin/env python3
"""
Verification script to check if Bhojpuri data collection setup is complete.
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    required_files = [
        'collect_data/__init__.py',
        'collect_data/scraper.py',
        'collect_data/data_cleaner.py',
        'collect_data/pipeline.py',
        'collect_data/run_pipeline.sh',
        'collect_data/README.md',
        'run_collection.py',
        'QUICKSTART.md',
    ]

    print("=" * 70)
    print("BHOJPURI DATA COLLECTION SETUP VERIFICATION")
    print("=" * 70)

    base_dir = Path(__file__).parent
    all_exist = True

    print("\nChecking required files:")
    print("-" * 70)

    for file_path in required_files:
        full_path = base_dir / file_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False

    print("-" * 70)

    if all_exist:
        print("✓ All files created successfully!")
    else:
        print("✗ Some files are missing!")
        return False

    # Check data directory
    print("\nChecking data directories:")
    print("-" * 70)

    data_dirs = ['data', 'data/raw', 'data/cleaned', 'data/processed']

    for dir_name in data_dirs:
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}/")

    print("-" * 70)

    # Check imports
    print("\nChecking Python imports:")
    print("-" * 70)

    try:
        sys.path.insert(0, str(base_dir))
        from collect_data.scraper import BhojpuriTextScraper
        print("  ✓ BhojpuriTextScraper imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import BhojpuriTextScraper: {e}")
        return False

    try:
        from collect_data.data_cleaner import BhojpuriDataCleaner
        print("  ✓ BhojpuriDataCleaner imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import BhojpuriDataCleaner: {e}")
        return False

    try:
        from collect_data.pipeline import BhojpuriDataPipeline
        print("  ✓ BhojpuriDataPipeline imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import BhojpuriDataPipeline: {e}")
        return False

    print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 70)
    print("\n✓ Setup is complete! Ready to collect Bhojpuri data.")
    print("\nNext steps:")
    print("  1. Read QUICKSTART.md for instructions")
    print("  2. Run: python3 run_collection.py --mode full")
    print("  3. Wait for pipeline to complete (2-3 hours)")
    print("  4. Check data/bhoj.txt for your training data")
    print("\n" + "=" * 70 + "\n")

    return True

if __name__ == '__main__':
    success = check_files()
    sys.exit(0 if success else 1)
