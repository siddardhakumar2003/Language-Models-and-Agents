#!/usr/bin/env python3
"""
Convenience script to run Bhojpuri data collection pipeline from bhojpuri/ root.

Usage:
    python3 run_collection.py                    # Full pipeline
    python3 run_collection.py --mode scrape      # Scrape only
    python3 run_collection.py --mode clean       # Clean only
    python3 run_collection.py --mode merge       # Merge only
    python3 run_collection.py --mode stats       # Show stats
"""

import sys
import os

# Add collect_data to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from collect_data.pipeline import BhojpuriDataPipeline

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Bhojpuri Data Collection Pipeline (Root Runner)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_collection.py                 # Run full pipeline
  python3 run_collection.py --mode scrape   # Scrape only
  python3 run_collection.py --mode clean    # Clean only
  python3 run_collection.py --mode stats    # Show statistics
  python3 run_collection.py --mode merge    # Merge to single file
        """
    )

    parser.add_argument(
        '--mode',
        choices=['scrape', 'clean', 'stats', 'full', 'merge'],
        default='full',
        help='Pipeline mode (default: full)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens to collect'
    )

    args = parser.parse_args()

    # Initialize pipeline with bhojpuri/data directory
    pipeline = BhojpuriDataPipeline(data_dir='./data')

    if args.mode == 'scrape':
        pipeline.run_scraper(max_tokens=args.max_tokens)
    elif args.mode == 'clean':
        pipeline.run_cleaner()
    elif args.mode == 'stats':
        pipeline.show_statistics()
    elif args.mode == 'merge':
        pipeline.merge_to_single_file()
    elif args.mode == 'full':
        pipeline.run_full_pipeline()

if __name__ == '__main__':
    main()
