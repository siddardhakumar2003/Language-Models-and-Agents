#!/usr/bin/env python3
"""
Main orchestration script for Telugu text data collection and processing pipeline.

Usage:
    python pipeline.py --mode scrape           # Run web scraper
    python pipeline.py --mode clean            # Clean collected data
    python pipeline.py --mode stats            # Show statistics
    python pipeline.py --mode full             # Run full pipeline
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Imports now in same directory (telugu/data_collect/)
from .enhanced_scraper import EnhancedTeluguScraper
from .data_cleaner import TeluguDataCleaner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, data_dir: str = None):
        # If no data_dir specified, use telugu/data/ (script location aware)
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.config_file = self.data_dir / "config.json"
        self.stats_file = self.data_dir / "stats.json"

        self.load_config()

    def load_config(self):
        """Load or create pipeline configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'target_tokens': 500_000_000,
                'batch_size': 100,
                'created_at': datetime.now().isoformat(),
            }
            self.save_config()

    def save_config(self):
        """Save pipeline configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def run_scraper(self, max_tokens: int = None):
        """Run web scraper (uses EnhancedTeluguScraper with API-driven discovery)"""
        logger.info("Starting scraper...")

        if max_tokens is None:
            max_tokens = self.config['target_tokens']

        scraper = EnhancedTeluguScraper(output_dir=str(self.data_dir))
        scraper.run_scraper_continuous(
            max_tokens=max_tokens,
            batch_size=self.config['batch_size']
        )

    def run_cleaner(self):
        """Run data cleaner"""
        logger.info("Starting data cleaner...")

        cleaner = TeluguDataCleaner(
            input_dir=str(self.data_dir / "raw"),
            output_dir=str(self.data_dir / "cleaned")
        )
        cleaner.run_cleaning()

    def merge_to_single_file(self, output_filename: str = "telugu.txt"):
        """Merge cleaned texts to a single file"""
        logger.info("Merging cleaned texts to single file...")

        cleaned_dir = self.data_dir / "cleaned"
        output_file = self.data_dir / output_filename

        if not cleaned_dir.exists():
            logger.error(f"Cleaned directory not found: {cleaned_dir}")
            return

        cleaned_files = sorted(cleaned_dir.glob("cleaned_*.jsonl"))
        if not cleaned_files:
            logger.warning("No cleaned JSONL files found")
            return

        total_lines = 0
        with open(output_file, 'w', encoding='utf-8') as out:
            for cleaned_file in cleaned_files:
                try:
                    with open(cleaned_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            text = data.get('text', '').strip()
                            if text:
                                out.write(text + '\n')
                                total_lines += 1
                except Exception as e:
                    logger.warning(f"Error reading {cleaned_file}: {e}")

        logger.info(f"Merged {total_lines} lines to {output_file}")
        return total_lines

    def show_statistics(self):
        """Display pipeline statistics"""
        logger.info("Pipeline Statistics:")
        logger.info(f"{'='*70}")

        raw_dir = self.data_dir / "raw"
        cleaned_dir = self.data_dir / "cleaned"

        # Count raw files
        raw_files = list(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
        raw_size = sum(f.stat().st_size for f in raw_files) if raw_files else 0

        # Count cleaned files
        cleaned_files = list(cleaned_dir.glob("cleaned_*.jsonl")) if cleaned_dir.exists() else []
        cleaned_size = sum(f.stat().st_size for f in cleaned_files) if cleaned_files else 0

        # Calculate tokens
        raw_tokens = self._estimate_tokens_from_size(raw_size)
        cleaned_tokens = self._count_tokens_from_jsonl(cleaned_files)

        logger.info(f"\nRAW DATA (unfiltered):")
        logger.info(f"  Files: {len(raw_files)}")
        logger.info(f"  Size: {raw_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Estimated Tokens: {raw_tokens / 1e6:.2f}M")

        logger.info(f"\nCLEANED DATA (SLM-ready):")
        logger.info(f"  Files: {len(cleaned_files)}")
        logger.info(f"  Size: {cleaned_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Tokens: {cleaned_tokens / 1e6:.2f}M")

        progress_pct = (cleaned_tokens / self.config['target_tokens']) * 100
        logger.info(f"\nPROGRESS TO TARGET:")
        logger.info(f"  Target: {self.config['target_tokens'] / 1e6:.0f}M tokens")
        logger.info(f"  Current (cleaned): {cleaned_tokens / 1e6:.2f}M tokens")
        logger.info(f"  Progress: {progress_pct:.1f}%")

        logger.info(f"{'='*70}\n")

    def _estimate_tokens_from_size(self, size_bytes: int) -> int:
        """Estimate tokens from file size (rough: 4 chars per token)"""
        return size_bytes // 4

    def _count_tokens_from_jsonl(self, files: list) -> int:
        """Count actual tokens from JSONL files"""
        total_tokens = 0
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        total_tokens += data.get('tokens', 0)
            except Exception as e:
                logger.warning(f"Error reading {file}: {e}")

        return total_tokens

    def run_full_pipeline(self):
        """Run complete pipeline: scrape -> clean -> stats"""
        logger.info("Starting full pipeline...")
        logger.info(f"Target: {self.config['target_tokens'] / 1e6:.0f}M tokens")

        # Step 1: Scrape
        logger.info("\nStep 1: Web Scraping")
        logger.info(f"{'-'*70}")
        self.run_scraper(use_enhanced=True)

        # Step 2: Clean
        logger.info("\nStep 2: Data Cleaning")
        logger.info(f"{'-'*70}")
        self.run_cleaner()

        # Step 3: Statistics
        logger.info("\nStep 3: Final Statistics")
        logger.info(f"{'-'*70}")
        self.show_statistics()

        logger.info("\nPipeline completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Telugu Text Data Collection and Processing Pipeline"
    )
    parser.add_argument(
        '--mode',
        choices=['scrape', 'clean', 'merge', 'stats', 'full'],
        default='full',
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Data directory'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens to collect (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing'
    )

    args = parser.parse_args()

    pipeline = DataPipeline(data_dir=args.data_dir)

    if args.mode == 'scrape':
        pipeline.run_scraper(max_tokens=args.max_tokens)
    elif args.mode == 'clean':
        pipeline.run_cleaner()
    elif args.mode == 'merge':
        pipeline.merge_to_single_file()
    elif args.mode == 'stats':
        pipeline.show_statistics()
    elif args.mode == 'full':
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()
