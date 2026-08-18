#!/usr/bin/env python3
"""
Download Bhojpuri texts from archive.org with genuine Bhojpuri-specific search queries.

This script extends BhojpuriArchiveOrgDownloader with Bhojpuri-targeted search strategies
(instead of the default Hindi-proxy searches). It downloads both pre-extracted djvu.txt
and OCR-extracted texts from archive.org books tagged as Bhojpuri or Bhojpuri literature.

MANUAL STEP NOTE:
For items that are scanned-image-only PDFs (no _djvu.txt), manual curation/download is
required — the archive.org API search doesn't distinguish availability of djvu.txt vs
image-only items. These PDFs are then processed via download_archive_org_ocr_books.py
(real Tesseract OCR).

Usage:
    python3 -m bhojpuri.data_collect.download_archive_org_bhojpuri [--data-dir DATA_DIR] [--target-items N]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .archive_org_downloader import BhojpuriArchiveOrgDownloader
from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BhojpuriArchiveOrgBhojpuriDownloader(BhojpuriArchiveOrgDownloader):
    """Extended downloader targeting genuine Bhojpuri books from archive.org."""

    # Bhojpuri-specific search strategies (not Hindi proxy)
    SEARCH_STRATEGIES_BHOJPURI = [
        "language:(Bhojpuri) AND mediatype:(texts)",  # Explicit Bhojpuri language tag
        "subject:(Bhojpuri) AND mediatype:(texts)",   # Bhojpuri subject
        "title:(Bhojpuri OR bhojpuri) AND mediatype:(texts)",  # Bhojpuri in title
        "subject:(Bhojpuri literature OR Bhojpuri language) AND mediatype:(texts)",  # Literature
        "description:(Bhojpuri) AND mediatype:(texts)",  # Bhojpuri in description
    ]

    def __init__(self, output_dir=None, state_file=None, batch_prefix="arch_bho", **kwargs):
        """Initialize with Bhojpuri-specific search strategies."""
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent.parent / "data" / "archive_org_bhojpuri_raw")
        if state_file is None:
            state_file = str(Path(__file__).resolve().parent.parent / "data" / "archive_org_bhojpuri_state.json")

        super().__init__(
            language_query="Bhojpuri",
            script_range=(0x0900, 0x097F),  # Devanagari
            output_dir=output_dir,
            state_file=state_file,
            batch_prefix=batch_prefix,
            **kwargs
        )

        # Override with Bhojpuri-specific strategies
        self.SEARCH_STRATEGIES = self.SEARCH_STRATEGIES_BHOJPURI


def process_archive_org_bhojpuri(data_dir: Path, target_items: int = 500) -> Dict[str, int]:
    """Download, extract, clean, deduplicate, and merge Bhojpuri texts from archive.org."""
    logger.info("=" * 70)
    logger.info("ARCHIVE.ORG BHOJPURI TEXTS PROCESSING (Pre-extracted djvu.txt)")
    logger.info("=" * 70)
    logger.info(f"Target items: {target_items}")

    # Initialize downloader with Bhojpuri search strategies
    downloader = BhojpuriArchiveOrgBhojpuriDownloader(
        output_dir=str(data_dir / "archive_org_bhojpuri_raw"),
        state_file=str(data_dir / "archive_org_bhojpuri_state.json")
    )

    logger.info(f"Searching archive.org for Bhojpuri texts (targeting {target_items} items)...")

    # Download up to target_items
    items_downloaded = 0
    for identifier in downloader.iter_new_identifiers():
        if items_downloaded >= target_items:
            logger.info(f"Reached target of {target_items} items")
            break

        logger.info(f"Processing [{items_downloaded + 1}/{target_items}] {identifier}...")
        downloaded_texts = downloader.download_texts_for_identifier(identifier, batch_size=50)

        if downloaded_texts > 0:
            items_downloaded += 1
            logger.info(f"  ✓ Downloaded {downloaded_texts} text chunks from {identifier}")
        else:
            logger.info(f"  ✗ No texts downloaded from {identifier}")

        # Save state periodically
        if items_downloaded % 10 == 0:
            downloader._save_state()

    downloader._save_state()
    logger.info(f"Downloaded from {items_downloaded} archive.org items")

    # Create cleaned directory
    raw_dir = data_dir / "archive_org_bhojpuri_raw"
    cleaned_dir = data_dir / "archive_org_bhojpuri_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    if not raw_dir.exists() or not list(raw_dir.glob("*.jsonl")):
        logger.warning("No raw texts found; skipping cleaning/merge")
        logger.info("=" * 70)
        return {"items_downloaded": items_downloaded, "texts_cleaned": 0}

    # Clean texts
    logger.info("Cleaning archive.org Bhojpuri texts...")
    cleaner = BhojpuriDataCleaner(
        input_dir=str(raw_dir),
        output_dir=str(cleaned_dir)
    )
    cleaner.run_cleaning()

    # Deduplicate against existing corpus
    logger.info("Deduplicating against existing corpus...")
    deduplicate_with_existing_corpus(
        data_dir=data_dir,
        source_cleaned_dir=cleaned_dir,
        output_cleaned_dir=cleaned_dir,
        state_file=None
    )

    # Merge into splits
    logger.info("Merging into train/val/test splits...")
    stats = merge_ocr_into_splits(
        data_dir=data_dir,
        accumulator_filename="bhoj.txt",
        split_filename="bhoj.txt",
        cleaned_dir_name="archive_org_bhojpuri_cleaned",
        temp_batch_filename="archive_org_bhojpuri_new_batch.txt"
    )

    # Archive cleaned dir
    import shutil
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"archive_org_bhojpuri_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"Archive.org Bhojpuri processing complete: {stats}")
    logger.info("=" * 70)

    return {**stats, "items_downloaded": items_downloaded}


def main():
    parser = argparse.ArgumentParser(description="Download Bhojpuri texts from archive.org")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    parser.add_argument(
        "--target-items",
        type=int,
        default=500,
        help="Target number of archive.org items to download (default: 500)"
    )
    args = parser.parse_args()

    process_archive_org_bhojpuri(args.data_dir, target_items=args.target_items)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    archive_source = "archive.org (Bhojpuri-tagged books, pre-extracted djvu.txt)"
    if archive_source not in config.get("data_sources", []):
        config["data_sources"].append(archive_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
