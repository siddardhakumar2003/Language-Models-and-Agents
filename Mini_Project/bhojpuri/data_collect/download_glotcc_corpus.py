#!/usr/bin/env python3
"""
Download and process GlotCC-V1 bho-Deva corpus, clean, deduplicate, and merge.

Downloads the bho-Deva parquet subset from GlotCC via direct HTTPS,
runs through BhojpuriDataCleaner, deduplicates against existing corpus,
and appends to train/val/test splits.

Usage:
    python3 -m bhojpuri.data_collect.download_glotcc_corpus [--data-dir DATA_DIR]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import shutil

import pandas as pd
import requests
from tqdm import tqdm

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GlotCC-V1 bho-Deva parquet URL
GLOTCC_BHO_PARQUET = "https://huggingface.co/datasets/cis-lmu/GlotCC-V1/resolve/main/v1.0/bho-Deva/bho-Deva_0.parquet"


def download_glotcc_parquet(data_dir: Path) -> Path:
    """Download the GlotCC-V1 bho-Deva parquet file."""
    download_dir = data_dir / "glotcc_raw_download"
    download_dir.mkdir(exist_ok=True)

    parquet_file = download_dir / "glotcc_bho_Deva.parquet"

    if parquet_file.exists():
        logger.info(f"Parquet file already exists at {parquet_file}")
        return parquet_file

    logger.info(f"Downloading GlotCC-V1 bho-Deva parquet from {GLOTCC_BHO_PARQUET}")

    try:
        response = requests.get(GLOTCC_BHO_PARQUET, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192

        with open(parquet_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded parquet file: {parquet_file.stat().st_size / 1024 / 1024:.1f} MB")
        return parquet_file

    except Exception as e:
        raise RuntimeError(f"Failed to download GlotCC-V1 parquet: {e}")


def parquet_to_jsonl(parquet_file: Path, raw_dir: Path) -> int:
    """Convert parquet to JSONL. Note: GlotCC uses 'content' field, not 'text'."""
    logger.info(f"Reading parquet file: {parquet_file}")

    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file: {e}")

    if 'content' not in df.columns:
        raise ValueError("Parquet file does not have a 'content' column")

    logger.info(f"Parquet file contains {len(df)} rows")

    # Write to JSONL in batches
    batch_size = 5000
    batch_num = 0
    text_count = 0

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch_file = raw_dir / f"glotcc_batch_{batch_num:05d}.jsonl"

        with open(batch_file, 'w', encoding='utf-8') as f:
            for _, row in batch.iterrows():
                # Skip rows with quality warnings (check for non-empty array/list)
                qw = row.get('quality-warnings')
                if qw is not None and (isinstance(qw, list) and len(qw) > 0):
                    continue

                # GlotCC uses 'content' field; remap to 'text' for cleaner
                text = str(row.get('content', '')).strip()
                if text:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    text_count += 1

        logger.info(f"Wrote {text_count - (batch_size * batch_num if batch_num > 0 else 0)} entries to {batch_file.name}")
        batch_num += 1

    logger.info(f"Converted {text_count} texts from parquet to JSONL")
    return text_count


def process_glotcc_corpus(data_dir: Path) -> Dict[str, int]:
    """Download, parse, clean, deduplicate, and merge GlotCC-V1 bho-Deva corpus."""
    logger.info("=" * 70)
    logger.info("GLOTCC-V1 BHO-DEVA CORPUS PROCESSING")
    logger.info("=" * 70)

    # Download parquet
    parquet_file = download_glotcc_parquet(data_dir)

    # Create raw and cleaned directories
    raw_dir = data_dir / "glotcc_raw"
    cleaned_dir = data_dir / "glotcc_cleaned"
    raw_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    # Convert parquet to JSONL
    text_count = parquet_to_jsonl(parquet_file, raw_dir)

    if text_count == 0:
        logger.warning("No texts extracted from parquet; skipping cleaning/merge")
        return {"texts_extracted": 0}

    # Clean texts
    logger.info("Cleaning GlotCC texts...")
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
        cleaned_dir_name="glotcc_cleaned",
        temp_batch_filename="glotcc_new_batch.txt"
    )

    # Clean up
    logger.info("Cleaning up intermediate files...")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if (data_dir / "glotcc_raw_download").exists():
        shutil.rmtree(data_dir / "glotcc_raw_download")

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"glotcc_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"GlotCC corpus processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process GlotCC-V1 bho-Deva corpus")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_glotcc_corpus(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    glotcc_source = "GlotCC-V1 (bho-Deva, web-crawl, 949 docs)"
    if glotcc_source not in config.get("data_sources", []):
        config["data_sources"].append(glotcc_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
