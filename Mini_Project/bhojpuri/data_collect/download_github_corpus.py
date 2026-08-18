#!/usr/bin/env python3
"""
Download and process fossdot/bhojpuri-corpus, clean, deduplicate, and merge into splits.

Downloads the GitHub repo, extracts reviewed JSONL entries, runs through
BhojpuriDataCleaner, deduplicates against existing corpus, and appends to splits.

Usage:
    python3 -m bhojpuri.data_collect.download_github_corpus [--data-dir DATA_DIR]
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

GITHUB_REPO = "https://github.com/fossdot/bhojpuri-corpus.git"


def download_github_repo(data_dir: Path) -> Path:
    """Clone the fossdot/bhojpuri-corpus GitHub repository."""
    repo_dir = data_dir / "fossdot_bhojpuri_corpus_repo"

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


def extract_reviewed_texts(repo_dir: Path) -> Dict[str, int]:
    """Extract reviewed texts from fossdot corpus JSONL files."""
    data_dir = repo_dir / "data" / "reviewed"
    raw_dir = Path(str(Path(__file__).parent.parent / "data")) / "fossdot_raw"
    raw_dir.mkdir(exist_ok=True)

    logger.info(f"Extracting reviewed texts from {data_dir}")

    if not data_dir.exists():
        logger.warning(f"Reviewed data directory not found: {data_dir}")
        return {"reviewed_count": 0, "rejected_count": 0}

    text_count = 0
    rejected_count = 0
    batch_num = 0
    batch_texts = []
    batch_size = 500

    # Iterate over all JSONL files in data/reviewed/
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        logger.info(f"Processing {jsonl_file.name}...")

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        rejected_count += 1
                        continue

                    # Filter to reviewed entries only
                    if entry.get("status") != "completed" and entry.get("reviewed") != True:
                        rejected_count += 1
                        continue

                    text = entry.get("text", "").strip()
                    if not text or len(text) < 50:
                        rejected_count += 1
                        continue

                    batch_texts.append({"text": text})
                    text_count += 1

                    if len(batch_texts) >= batch_size:
                        # Write batch
                        batch_file = raw_dir / f"fossdot_batch_{batch_num:05d}.jsonl"
                        with open(batch_file, 'w', encoding='utf-8') as out:
                            for entry in batch_texts:
                                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        batch_num += 1
                        batch_texts = []

        except Exception as e:
            logger.error(f"Error processing {jsonl_file}: {e}")
            continue

    # Write final batch
    if batch_texts:
        batch_file = raw_dir / f"fossdot_batch_{batch_num:05d}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as out:
            for entry in batch_texts:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Extracted {text_count} reviewed texts ({rejected_count} rejected)")
    return {"reviewed_count": text_count, "rejected_count": rejected_count, "raw_dir": raw_dir}


def process_github_corpus(data_dir: Path) -> Dict[str, int]:
    """Download, parse, clean, deduplicate, and merge fossdot corpus."""
    logger.info("=" * 70)
    logger.info("FOSSDOT BHOJPURI CORPUS PROCESSING")
    logger.info("=" * 70)

    # Download repo
    repo_dir = download_github_repo(data_dir)

    # Extract reviewed texts
    extraction_stats = extract_reviewed_texts(repo_dir)
    if extraction_stats["reviewed_count"] == 0:
        logger.warning("No reviewed texts extracted; skipping cleaning/merge")
        return {"texts_extracted": 0}

    raw_dir = extraction_stats["raw_dir"]
    cleaned_dir = data_dir / "fossdot_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    # Clean texts
    logger.info("Cleaning fossdot corpus texts...")
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
        cleaned_dir_name="fossdot_cleaned",
        temp_batch_filename="fossdot_new_batch.txt"
    )

    # Clean up
    logger.info("Cleaning up intermediate files...")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"fossdot_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"Fossdot corpus processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process fossdot/bhojpuri-corpus")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_github_corpus(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    fossdot_source = "fossdot/bhojpuri-corpus (GitHub: reviewed community-contributed texts, ~171 sentences)"
    if fossdot_source not in config.get("data_sources", []):
        config["data_sources"].append(fossdot_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
