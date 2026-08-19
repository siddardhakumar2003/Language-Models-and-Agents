#!/usr/bin/env python3
"""Download and merge goldfish-models/fish-food bho_deva (262K rows) into corpus."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import shutil

from datasets import load_dataset

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fishfood_bho() -> Dict:
    """Load fish-food bho_deva (262K rows) via HF datasets."""
    logger.info("Loading fish-food bho_deva (262K rows)...")
    try:
        ds = load_dataset("goldfish-models/fish-food", "bho_deva", split="train", trust_remote_code=True)
        logger.info(f"Loaded {len(ds)} rows from fish-food bho_deva")
        return ds
    except Exception as e:
        raise RuntimeError(f"Failed to load fish-food dataset: {e}")


def process_fishfood_corpus(data_dir: Path) -> Dict[str, int]:
    """Load, clean, deduplicate, and merge fish-food bho_deva corpus."""
    logger.info("=" * 70)
    logger.info("FISH-FOOD BHO_DEVA CORPUS PROCESSING (262K rows)")
    logger.info("=" * 70)

    # Create raw and cleaned directories
    raw_dir = data_dir / "fishfood_raw"
    cleaned_dir = data_dir / "fishfood_cleaned"
    raw_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    # Load dataset and flatten to JSONL
    ds = load_fishfood_bho()

    logger.info(f"Converting {len(ds)} rows to JSONL...")
    text_count = 0
    with open(raw_dir / "fishfood_batch_00000.jsonl", 'w', encoding='utf-8') as f:
        for i, record in enumerate(ds):
            text = record.get('text', '').strip()
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                text_count += 1
            if (i + 1) % 50000 == 0:
                logger.info(f"  Processed {i+1}/{len(ds)} rows...")

    logger.info(f"Converted {text_count} texts to JSONL")

    if text_count == 0:
        logger.warning("No texts extracted; skipping cleaning/merge")
        return {"texts_extracted": 0}

    # Clean texts
    logger.info("Cleaning fish-food texts...")
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
        cleaned_dir_name="fishfood_cleaned",
        temp_batch_filename="fishfood_new_batch.txt"
    )

    # Clean up
    logger.info("Cleaning up intermediate files...")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"fishfood_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"Fish-food corpus processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process fish-food bho_deva corpus")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_fishfood_corpus(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    fishfood_source = "goldfish-models/fish-food (bho_deva, 262K rows)"
    if fishfood_source not in config.get("data_sources", []):
        config["data_sources"].append(fishfood_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
