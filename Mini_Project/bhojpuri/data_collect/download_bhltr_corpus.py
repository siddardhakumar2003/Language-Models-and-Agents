#!/usr/bin/env python3
"""
Download and process BHLTR (Bhojpuri Language Resources) corpus, clean, deduplicate, and merge.

Downloads github.com/shashwatup9k/bho-resources, extracts .bho monolingual texts,
runs through BhojpuriDataCleaner, deduplicates against existing corpus,
and appends to train/val/test splits.

License: CC BY-NC-SA 4.0 (non-commercial, fine for academic projects)

Usage:
    python3 -m bhojpuri.data_collect.download_bhltr_corpus [--data-dir DATA_DIR]
"""

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GITHUB_REPO = "https://github.com/shashwatup9k/bho-resources.git"


def download_github_repo(data_dir: Path) -> Path:
    """Clone the shashwatup9k/bho-resources GitHub repository."""
    repo_dir = data_dir / "bhltr_bho_resources_repo"

    if repo_dir.exists():
        logger.info(f"Repository already cloned at {repo_dir}")
        return repo_dir

    logger.info(f"Cloning {GITHUB_REPO}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", GITHUB_REPO, str(repo_dir)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repository: {result.stderr}")

    logger.info(f"Repository cloned to {repo_dir}")
    return repo_dir


def extract_monolingual_texts(repo_dir: Path) -> Dict[str, int]:
    """Extract monolingual .bho texts from BHLTR corpus."""
    mono_dir = repo_dir / "mono-bho-corpus"
    raw_dir = Path(str(Path(__file__).parent.parent / "data")) / "bhltr_raw"
    raw_dir.mkdir(exist_ok=True)

    logger.info(f"Extracting monolingual texts from {mono_dir}")

    if not mono_dir.exists():
        logger.warning(f"Monolingual corpus directory not found: {mono_dir}")
        return {"text_count": 0}

    text_count = 0
    batch_num = 0
    batch_texts = []
    batch_size = 500

    # Find all .bho files in the monolingual corpus
    for bho_file in sorted(mono_dir.rglob("*.bho")):
        logger.info(f"Processing {bho_file.relative_to(mono_dir)}...")

        try:
            with open(bho_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    text = line.strip()
                    if not text or len(text) < 50:
                        continue

                    batch_texts.append({"text": text})
                    text_count += 1

                    if len(batch_texts) >= batch_size:
                        # Write batch
                        batch_file = raw_dir / f"bhltr_batch_{batch_num:05d}.jsonl"
                        with open(batch_file, 'w', encoding='utf-8') as out:
                            for entry in batch_texts:
                                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        batch_num += 1
                        batch_texts = []

        except Exception as e:
            logger.error(f"Error processing {bho_file}: {e}")
            continue

    # Write final batch
    if batch_texts:
        batch_file = raw_dir / f"bhltr_batch_{batch_num:05d}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as out:
            for entry in batch_texts:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Extracted {text_count} monolingual texts from .bho files")
    return {"text_count": text_count, "raw_dir": raw_dir}


def process_bhltr_corpus(data_dir: Path) -> Dict[str, int]:
    """Download, parse, clean, deduplicate, and merge BHLTR corpus."""
    logger.info("=" * 70)
    logger.info("BHLTR BHOJPURI CORPUS PROCESSING (CC BY-NC-SA 4.0)")
    logger.info("=" * 70)

    # Download repo
    repo_dir = download_github_repo(data_dir)

    # Extract monolingual texts
    extraction_stats = extract_monolingual_texts(repo_dir)
    if extraction_stats["text_count"] == 0:
        logger.warning("No texts extracted; skipping cleaning/merge")
        return {"texts_extracted": 0}

    raw_dir = extraction_stats["raw_dir"]
    cleaned_dir = data_dir / "bhltr_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    # Clean texts
    logger.info("Cleaning BHLTR corpus texts...")
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
        cleaned_dir_name="bhltr_cleaned",
        temp_batch_filename="bhltr_new_batch.txt"
    )

    # Clean up
    logger.info("Cleaning up intermediate files...")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"bhltr_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"BHLTR corpus processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process BHLTR Bhojpuri Language Resources")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_bhltr_corpus(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    bhltr_source = "BHLTR (shashwatup9k/bho-resources, monolingual corpus, CC BY-NC-SA 4.0)"
    if bhltr_source not in config.get("data_sources", []):
        config["data_sources"].append(bhltr_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
