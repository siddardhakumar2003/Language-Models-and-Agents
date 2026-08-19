#!/usr/bin/env python3
"""
Download and process HPLT3.0 bho_Deva corpus, clean, deduplicate, and merge.

Downloads the bho_Deva shards from HPLT via the map file,
runs through BhojpuriDataCleaner, deduplicates against existing corpus,
and appends to train/val/test splits.

Usage:
    python3 -m bhojpuri.data_collect.download_hplt_corpus [--data-dir DATA_DIR]
"""

import argparse
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
import shutil

import requests
from tqdm import tqdm
import zstandard as zstd

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HPLT_MAP_URL = "https://data.hplt-project.org/three/sorted/bho_Deva.map"


def download_hplt_shards(data_dir: Path) -> list:
    """Download the HPLT3.0 bho_Deva shards via the map file."""
    download_dir = data_dir / "hplt_raw_download"
    download_dir.mkdir(exist_ok=True)

    logger.info(f"Fetching HPLT map from {HPLT_MAP_URL}")
    try:
        response = requests.get(HPLT_MAP_URL, timeout=30)
        response.raise_for_status()
        shard_urls = response.text.strip().split('\n')
    except Exception as e:
        raise RuntimeError(f"Failed to fetch HPLT map: {e}")

    logger.info(f"Found {len(shard_urls)} shards in map file")

    downloaded_files = []
    for shard_url in shard_urls:
        shard_url = shard_url.strip()
        if not shard_url:
            continue

        filename = shard_url.split('/')[-1]
        local_path = download_dir / filename

        if local_path.exists():
            logger.info(f"Shard already exists: {filename}")
            downloaded_files.append(local_path)
            continue

        logger.info(f"Downloading shard: {filename}")
        try:
            response = requests.get(shard_url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192

            with open(local_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded: {filename} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
            downloaded_files.append(local_path)

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if local_path.exists():
                local_path.unlink()

    return downloaded_files


def decompress_and_convert_to_jsonl(download_dir: Path, raw_dir: Path) -> int:
    """Decompress JSONL.zst files and convert to plain JSONL."""
    raw_dir.mkdir(exist_ok=True)

    total_lines = 0
    batch_num = 0
    batch_lines = []
    batch_size = 5000

    for zst_file in sorted(download_dir.glob("*.jsonl.zst")):
        logger.info(f"Decompressing {zst_file.name}")

        try:
            with open(zst_file, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    for line in text_stream:
                        try:
                            record = json.loads(line)

                            if record.get('filter') != 'keep':
                                continue

                            text = record.get('text', '').strip()
                            if text:
                                batch_lines.append({"text": text})
                                total_lines += 1

                                if len(batch_lines) >= batch_size:
                                    batch_file = raw_dir / f"hplt_batch_{batch_num:05d}.jsonl"
                                    with open(batch_file, 'w', encoding='utf-8') as out:
                                        for item in batch_lines:
                                            out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                    logger.info(f"Wrote batch {batch_num}: {len(batch_lines)} records")
                                    batch_num += 1
                                    batch_lines = []

                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            logger.error(f"Failed to decompress {zst_file.name}: {e}")

    # Write remaining batch
    if batch_lines:
        batch_file = raw_dir / f"hplt_batch_{batch_num:05d}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as out:
            for item in batch_lines:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Wrote final batch {batch_num}: {len(batch_lines)} records")

    logger.info(f"Decompressed and converted {total_lines} lines to JSONL")
    return total_lines


def process_hplt_corpus(data_dir: Path) -> Dict[str, int]:
    """Download, decompress, clean, deduplicate, and merge HPLT bho_Deva corpus."""
    logger.info("=" * 70)
    logger.info("HPLT3.0 BHO_DEVA CORPUS PROCESSING")
    logger.info("=" * 70)

    # Download shards
    downloaded_files = download_hplt_shards(data_dir)

    if not downloaded_files:
        logger.warning("No shards downloaded; skipping conversion/merge")
        return {"texts_extracted": 0}

    # Create raw and cleaned directories
    raw_dir = data_dir / "hplt_raw"
    cleaned_dir = data_dir / "hplt_cleaned"
    raw_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    # Decompress and convert to JSONL
    text_count = decompress_and_convert_to_jsonl(data_dir / "hplt_raw_download", raw_dir)

    if text_count == 0:
        logger.warning("No texts extracted from shards; skipping cleaning/merge")
        return {"texts_extracted": 0}

    # Clean texts
    logger.info("Cleaning HPLT texts...")
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
        cleaned_dir_name="hplt_cleaned",
        temp_batch_filename="hplt_new_batch.txt"
    )

    # Clean up
    logger.info("Cleaning up intermediate files...")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if (data_dir / "hplt_raw_download").exists():
        shutil.rmtree(data_dir / "hplt_raw_download")

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"hplt_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"HPLT corpus processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process HPLT3.0 bho_Deva corpus")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_hplt_corpus(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    hplt_source = "HPLT3.0 (bho_Deva, web-crawl, ~32.8k docs)"
    if hplt_source not in config.get("data_sources", []):
        config["data_sources"].append(hplt_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
