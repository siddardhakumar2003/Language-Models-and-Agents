#!/usr/bin/env python3
"""
Merge OCR-cleaned texts into split datasets and accumulator file (Bhojpuri).

After ocr_extractor.py + existing BhojpuriDataCleaner produce ocr_cleaned/*.jsonl,
this script:
1. Reconstructs bhoj.txt from existing split fragments if absent
2. Flattens all ocr_cleaned/*.jsonl into a temp batch file
3. Splits the temp batch 80/10/10 using existing split_dataset.py logic
4. Appends the three new chunks onto existing train/val/test/bhoj.txt files
5. Appends the full new batch onto the accumulator bhoj.txt

Usage:
    python3 ocr_merge.py [--data-dir DATADIR]
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

from .split_dataset import split_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_accumulator_exists(data_dir: Path, filename: str) -> Path:
    """Reconstruct accumulator from split fragments if it doesn't exist."""
    accumulator = data_dir / filename
    if accumulator.exists():
        logger.info(f"{filename} already exists, skipping reconstruction")
        return accumulator

    logger.info(f"Reconstructing {filename} from split fragments...")
    with open(accumulator, 'w', encoding='utf-8') as out:
        for split in ('train', 'val', 'test'):
            frag = data_dir / split / filename
            if frag.exists():
                logger.info(f"  Appending {split}/{filename}")
                with open(frag, 'r', encoding='utf-8') as f:
                    shutil.copyfileobj(f, out)

    logger.info(f"Reconstructed {filename}")
    return accumulator


def flatten_cleaned_jsonl(cleaned_dir: Path, output_file: Path) -> int:
    """Flatten all cleaned_*.jsonl files into a single text file."""
    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for cleaned_jsonl in sorted(cleaned_dir.glob("cleaned_*.jsonl")):
            try:
                with open(cleaned_jsonl, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '').strip()
                            if text:
                                out.write(text + '\n')
                                total_lines += 1
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                logger.error(f"Error reading {cleaned_jsonl}: {e}")

    logger.info(f"Flattened {total_lines} lines to {output_file.name}")
    return total_lines


def merge_ocr_into_splits(
    data_dir: Path,
    accumulator_filename: str = "bhoj.txt",
    split_filename: str = "bhoj.txt",
    cleaned_dir_name: str = "ocr_cleaned",
    temp_batch_filename: str = "ocr_new_batch.txt",
) -> Dict:
    """
    Merge OCR data into splits and accumulator.

    Returns dict with merge stats:
    {
        'total_lines': int,
        'train_lines': int, 'val_lines': int, 'test_lines': int,
        'accumulator_before': int, 'accumulator_after': int,
    }
    """
    cleaned_dir = data_dir / cleaned_dir_name
    if not cleaned_dir.exists():
        logger.error(f"Cleaned directory not found: {cleaned_dir}")
        return {}

    logger.info("=" * 70)
    logger.info("OCR MERGE PROCESS (Bhojpuri)")
    logger.info("=" * 70)

    # Step 1: Ensure accumulator exists
    accumulator = ensure_accumulator_exists(data_dir, accumulator_filename)
    accumulator_lines_before = sum(1 for _ in open(accumulator, 'r', encoding='utf-8'))

    # Step 2: Flatten ocr_cleaned to temp batch file
    ocr_batch_file = data_dir / temp_batch_filename
    total_lines = flatten_cleaned_jsonl(cleaned_dir, ocr_batch_file)

    if total_lines == 0:
        logger.warning("No lines to merge, exiting")
        ocr_batch_file.unlink(missing_ok=True)
        return {'total_lines': 0, 'train_lines': 0, 'val_lines': 0, 'test_lines': 0}

    # Step 3: Split the temp batch file
    logger.info(f"\nSplitting {ocr_batch_file.name} into 80/10/10...")
    split_result = split_file(
        input_path=ocr_batch_file,
        output_dir=data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    if not split_result:
        logger.error("Split failed")
        ocr_batch_file.unlink(missing_ok=True)
        return {}

    train_lines = split_result.get('train_lines', 0)
    val_lines = split_result.get('val_lines', 0)
    test_lines = split_result.get('test_lines', 0)

    logger.info(f"\nSplit result: train={train_lines}, val={val_lines}, test={test_lines}")

    # Step 4: Append split chunks onto existing split files
    logger.info(f"\nAppending split chunks to existing split files...")
    for split in ('train', 'val', 'test'):
        src = data_dir / split / temp_batch_filename
        dst = data_dir / split / split_filename
        if not src.exists():
            logger.warning(f"{src} not found, skipping")
            continue

        logger.info(f"  Appending {split}/{temp_batch_filename} → {split}/{split_filename}")
        with open(src, 'r', encoding='utf-8') as fin, open(dst, 'a', encoding='utf-8') as fout:
            shutil.copyfileobj(fin, fout)
        src.unlink()

    # Step 5: Append full batch to accumulator
    logger.info(f"\nAppending full batch to accumulator...")
    with open(ocr_batch_file, 'r', encoding='utf-8') as fin, open(accumulator, 'a', encoding='utf-8') as fout:
        shutil.copyfileobj(fin, fout)
    ocr_batch_file.unlink()

    accumulator_lines_after = sum(1 for _ in open(accumulator, 'r', encoding='utf-8'))
    logger.info(f"Accumulator: {accumulator_lines_before} → {accumulator_lines_after} lines")

    # Step 6: Archive ocr_cleaned to prevent double-append
    archive_dir = data_dir / f"ocr_cleaned_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"\nArchiving {cleaned_dir.name} to {archive_dir.name}")
    shutil.move(str(cleaned_dir), str(archive_dir))

    logger.info("=" * 70)
    logger.info("Merge complete!")
    logger.info("=" * 70)

    return {
        'total_lines': total_lines,
        'train_lines': train_lines,
        'val_lines': val_lines,
        'test_lines': test_lines,
        'accumulator_before': accumulator_lines_before,
        'accumulator_after': accumulator_lines_after,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge OCR data into splits (Bhojpuri)")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help='Data directory (default: bhojpuri/data/)'
    )

    args = parser.parse_args()

    result = merge_ocr_into_splits(args.data_dir)

    if result:
        print(f"\n✓ Merge successful")
        print(f"  Total lines: {result['total_lines']}")
        print(f"  Train: {result['train_lines']}, Val: {result['val_lines']}, Test: {result['test_lines']}")
        return 0
    else:
        print(f"\n✗ Merge failed or no data to merge")
        return 1


if __name__ == "__main__":
    exit(main())
