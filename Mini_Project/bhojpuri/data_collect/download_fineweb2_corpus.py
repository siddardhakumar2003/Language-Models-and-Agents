#!/usr/bin/env python3
"""
Download and process HuggingFace fineweb-2 bho_Deva corpus, clean, deduplicate, and merge.

Downloads the bho_Deva parquet subset(s) from HuggingFaceFW/fineweb-2 via direct HTTPS,
runs through BhojpuriDataCleaner, deduplicates against existing corpus,
and appends to train/val/test splits.

Usage:
    python3 -m bhojpuri.data_collect.download_fineweb2_corpus [--data-dir DATA_DIR]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import shutil

import pandas as pd
from datasets import load_dataset

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fineweb2_bho_deva() -> pd.DataFrame:
    """Load fineweb-2 bho_Deva dataset via HuggingFace datasets library."""
    logger.info("Loading fineweb-2 bho_Deva via HuggingFace datasets API...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-2", "bho_Deva", split="train", trust_remote_code=True)
        df = ds.to_pandas()
        logger.info(f"Loaded {len(df)} rows from fineweb-2 bho_Deva")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load fineweb-2 dataset: {e}")


def parquet_to_jsonl(df: pd.DataFrame, raw_dir: Path) -> int:
    """Convert DataFrame to JSONL."""
    if 'text' not in df.columns:
        raise ValueError("DataFrame does not have a 'text' column")

    logger.info(f"DataFrame contains {len(df)} rows")

    # Write to JSONL in batches
    batch_size = 5000
    batch_num = 0
    text_count = 0

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch_file = raw_dir / f"fineweb2_batch_{batch_num:05d}.jsonl"

        with open(batch_file, 'w', encoding='utf-8') as f:
            for _, row in batch.iterrows():
                text = str(row.get('text', '')).strip()
                if text:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    text_count += 1

        logger.info(f"Wrote {len(batch)} entries to {batch_file.name}")
        batch_num += 1

    logger.info(f"Converted {text_count} texts from parquet to JSONL")
    return text_count


def process_fineweb2_corpus(data_dir: Path) -> Dict[str, int]:
    """Download, parse, clean, deduplicate, and merge fineweb-2 bho_Deva corpus."""
    logger.info("=" * 70)
    logger.info("FINEWEB-2 BHO_DEVA CORPUS PROCESSING")
    logger.info("=" * 70)

    # Create raw and cleaned directories
    raw_dir = data_dir / "fineweb2_raw"
    cleaned_dir = data_dir / "fineweb2_cleaned"
    raw_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    # Load dataset and convert to JSONL
    df = load_fineweb2_bho_deva()
    text_count = parquet_to_jsonl(df, raw_dir)

    if text_count == 0:
        logger.warning("No texts extracted from parquet; skipping cleaning/merge")
        return {"texts_extracted": 0}

    # Clean texts
    logger.info("Cleaning fineweb-2 texts...")
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
        cleaned_dir_name="fineweb2_cleaned",
        temp_batch_filename="fineweb2_new_batch.txt"
    )

    # Clean up
    logger.info("Cleaning up intermediate files...")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"fineweb2_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"FineWeb-2 corpus processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process HuggingFace fineweb-2 bho_Deva corpus")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_fineweb2_corpus(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    fineweb2_source = "HuggingFace fineweb-2 (bho_Deva subset, ~18.7k CommonCrawl-derived docs)"
    if fineweb2_source not in config.get("data_sources", []):
        config["data_sources"].append(fineweb2_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
