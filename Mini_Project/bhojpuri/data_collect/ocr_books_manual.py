#!/usr/bin/env python3
"""
Manual OCR of Bhojpuri book PDFs/images using Tesseract.

IMPORTANT: This is a MANUAL STEP. You must:
1. Download/curate Bhojpuri book PDFs from archive.org or other sources
2. Place them in bhojpuri/data_collect/ocr_sources/
3. Run this script to OCR them with Tesseract

The archive.org API doesn't distinguish between items with pre-OCR'd djvu.txt
and scanned-image-only items. To get genuine OCR data from scanned Bhojpuri books:
- Visit archive.org/search.php with "language:(Bhojpuri) AND mediatype:(texts)"
- Manually check which items are image-only (no "Read Online" / djvu.txt link)
- Download those PDFs to ocr_sources/

Usage:
    python3 -m bhojpuri.data_collect.ocr_books_manual [--data-dir DATA_DIR]

This script:
1. Looks for PDFs/images in bhojpuri/data_collect/ocr_sources/
2. Runs Tesseract OCR (lang='hin' for Devanagari)
3. Cleans the OCR output
4. Deduplicates against existing corpus
5. Merges into train/val/test splits
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import shutil

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

try:
    from .ocr_extractor import BhojpuriOCRExtractor
    ocr_available = True
except ImportError:
    ocr_available = False
    BhojpuriOCRExtractor = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_manual_ocr(data_dir: Path) -> Dict[str, int]:
    """Process manually-curated PDFs/images with Tesseract OCR."""
    logger.info("=" * 70)
    logger.info("MANUAL TESSERACT OCR OF BHOJPURI BOOK PDFs")
    logger.info("=" * 70)

    if not ocr_available:
        logger.error("OCR module not available!")
        logger.error("Required packages not installed: pytesseract, PyMuPDF")
        logger.error("To enable OCR, run: pip install pytesseract PyMuPDF")
        return {"pdfs_processed": 0, "texts_extracted": 0, "error": "OCR not available"}

    ocr_sources_dir = Path(__file__).resolve().parent / "ocr_sources"
    logger.info(f"Looking for PDFs/images in {ocr_sources_dir}")

    if not ocr_sources_dir.exists():
        logger.warning(f"ocr_sources directory not found: {ocr_sources_dir}")
        logger.info("Please manually download Bhojpuri book PDFs and place them in:")
        logger.info(f"  {ocr_sources_dir}/")
        logger.info("")
        logger.info("To curate Bhojpuri books from archive.org:")
        logger.info("1. Go to https://archive.org/search.php")
        logger.info("2. Search: language:(Bhojpuri) AND mediatype:(texts)")
        logger.info("3. Filter to items WITHOUT pre-OCR'd text (check if 'Read Online' is unavailable)")
        logger.info("4. Download PDFs and place in ocr_sources/")
        logger.info("")
        return {"pdfs_processed": 0, "texts_extracted": 0}

    # List source files
    pdf_files = sorted(ocr_sources_dir.glob("*.pdf"))
    image_files = sorted(ocr_sources_dir.glob("*.png")) + \
                  sorted(ocr_sources_dir.glob("*.jpg")) + \
                  sorted(ocr_sources_dir.glob("*.jpeg")) + \
                  sorted(ocr_sources_dir.glob("*.tiff")) + \
                  sorted(ocr_sources_dir.glob("*.tif"))

    all_files = pdf_files + image_files

    if not all_files:
        logger.warning(f"No PDFs or images found in {ocr_sources_dir}")
        logger.info("Please download Bhojpuri book PDFs and place them in the directory above.")
        return {"pdfs_processed": 0, "texts_extracted": 0}

    logger.info(f"Found {len(all_files)} files to OCR ({len(pdf_files)} PDFs, {len(image_files)} images)")

    # Initialize OCR extractor
    ocr_raw_dir = data_dir / "ocr_books_raw"
    extractor = BhojpuriOCRExtractor(
        source_dir=str(ocr_sources_dir),
        output_dir=str(ocr_raw_dir),
        state_file=str(data_dir / "ocr_books_state.json"),
        lang="hin",  # Hindi/Devanagari — works for Bhojpuri too
        dpi=300
    )

    logger.info(f"Starting Tesseract OCR (lang='hin', dpi=300)...")
    logger.info("This may take several minutes depending on the number and size of files...")

    try:
        texts_extracted = extractor.extract_texts()
    except ImportError as e:
        logger.error(f"OCR extraction failed: {e}")
        logger.error("Please install required packages: pip install pytesseract PyMuPDF")
        logger.error("Also ensure system Tesseract is installed: sudo apt-get install tesseract-ocr")
        return {"pdfs_processed": 0, "texts_extracted": 0}

    if texts_extracted == 0:
        logger.warning("No texts extracted from OCR; skipping cleaning/merge")
        return {"pdfs_processed": len(all_files), "texts_extracted": 0}

    logger.info(f"Extracted {texts_extracted} texts from OCR")

    # Create cleaned directory
    cleaned_dir = data_dir / "ocr_books_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    # Clean texts
    logger.info("Cleaning OCR output...")
    cleaner = BhojpuriDataCleaner(
        input_dir=str(ocr_raw_dir),
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
        cleaned_dir_name="ocr_books_cleaned",
        temp_batch_filename="ocr_books_new_batch.txt"
    )

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"ocr_books_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"Manual OCR processing complete: {stats}")
    logger.info("=" * 70)

    return {**stats, "pdfs_processed": len(all_files)}


def main():
    parser = argparse.ArgumentParser(description="Manual Tesseract OCR of Bhojpuri book PDFs")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_manual_ocr(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    ocr_source = "Manual OCR (Tesseract, Bhojpuri book PDFs from archive.org/other sources)"
    if ocr_source not in config.get("data_sources", []):
        config["data_sources"].append(ocr_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
