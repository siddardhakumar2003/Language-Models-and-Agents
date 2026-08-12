#!/usr/bin/env python3
"""
Split dataset into train/val/test by whole lines (80/10/10).

Supports both small files (in-memory shuffle) and large files (streaming).
Automatically selects strategy based on file size.

For large files (>200MB), uses constant-memory streaming with per-line
probabilistic routing to preserve order and memory efficiency.

Usage:
    python3 split_dataset.py --input data/telugu.txt --output-dir data
    python3 split_dataset.py --input data/te.txt --output-dir /media/ubuntu/buffer
"""

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_file(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stream_threshold_bytes: int = 200 * 1024 * 1024,
) -> Dict[str, int]:
    """
    Split a text file into train/val/test by whole lines.

    Args:
        input_path: Input file to split
        output_dir: Output directory (creates train/val/test subdirs)
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility
        stream_threshold_bytes: Use streaming if file > this size

    Returns:
        Dict with line counts per split
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return {}

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    input_size = input_path.stat().st_size
    filename = input_path.name

    # Check disk space before streaming
    try:
        free_bytes = shutil.disk_usage(output_dir).free
        if free_bytes < int(1.2 * input_size):
            logger.error(f"Insufficient disk space: need {1.2 * input_size / 1e9:.1f}GB, have {free_bytes / 1e9:.1f}GB")
            return {}
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")

    # Select strategy based on size
    if input_size > stream_threshold_bytes:
        logger.info(f"File size {input_size / 1e9:.2f}GB > {stream_threshold_bytes / 1e9:.1f}GB, using streaming split")
        return _split_streaming(
            input_path, train_dir / filename, val_dir / filename, test_dir / filename,
            train_ratio, val_ratio, seed
        )
    else:
        logger.info(f"File size {input_size / 1e6:.1f}MB < {stream_threshold_bytes / 1e6:.1f}MB, using in-memory split")
        return _split_in_memory(
            input_path, train_dir / filename, val_dir / filename, test_dir / filename,
            train_ratio, val_ratio, seed
        )


def _split_in_memory(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, int]:
    """In-memory split with shuffling for small files."""
    logger.info(f"Loading {input_path.name} into memory...")

    # Load all lines
    lines = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return {}

    if not lines:
        logger.error("File is empty")
        return {}

    logger.info(f"Loaded {len(lines)} non-empty lines")

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(lines)

    # Split
    n = len(lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    # Write splits
    try:
        with open(train_path, 'w', encoding='utf-8') as f:
            for line in train_lines:
                f.write(line + '\n')

        with open(val_path, 'w', encoding='utf-8') as f:
            for line in val_lines:
                f.write(line + '\n')

        with open(test_path, 'w', encoding='utf-8') as f:
            for line in test_lines:
                f.write(line + '\n')

        logger.info(f"Train: {len(train_lines)} lines ({len(train_lines)/n*100:.1f}%)")
        logger.info(f"Val:   {len(val_lines)} lines ({len(val_lines)/n*100:.1f}%)")
        logger.info(f"Test:  {len(test_lines)} lines ({len(test_lines)/n*100:.1f}%)")

        return {
            'total_lines': n,
            'train_lines': len(train_lines),
            'val_lines': len(val_lines),
            'test_lines': len(test_lines),
            'strategy': 'in_memory'
        }

    except Exception as e:
        logger.error(f"Error writing splits: {e}")
        return {}


def _split_streaming(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, int]:
    """Streaming split for large files. Preserves order within each split."""
    logger.info(f"Streaming split of {input_path.name} (order-preserving)...")

    rng = random.Random(seed)

    train_count = 0
    val_count = 0
    test_count = 0

    try:
        with open(input_path, 'r', encoding='utf-8', buffering=1024*1024) as f_in:
            with open(train_path, 'w', encoding='utf-8', buffering=1024*1024) as f_train:
                with open(val_path, 'w', encoding='utf-8', buffering=1024*1024) as f_val:
                    with open(test_path, 'w', encoding='utf-8', buffering=1024*1024) as f_test:

                        for line in f_in:
                            line = line.rstrip('\n')
                            if not line.strip():
                                continue

                            # Probabilistic routing
                            r = rng.random()
                            if r < train_ratio:
                                f_train.write(line + '\n')
                                train_count += 1
                            elif r < train_ratio + val_ratio:
                                f_val.write(line + '\n')
                                val_count += 1
                            else:
                                f_test.write(line + '\n')
                                test_count += 1

                            # Progress logging
                            total = train_count + val_count + test_count
                            if total % 1_000_000 == 0:
                                logger.info(f"Processed {total / 1e6:.1f}M lines")

        total = train_count + val_count + test_count
        logger.info(f"Train: {train_count} lines ({train_count/total*100:.1f}%)")
        logger.info(f"Val:   {val_count} lines ({val_count/total*100:.1f}%)")
        logger.info(f"Test:  {test_count} lines ({test_count/total*100:.1f}%)")

        return {
            'total_lines': total,
            'train_lines': train_count,
            'val_lines': val_count,
            'test_lines': test_count,
            'strategy': 'streaming_order_preserved'
        }

    except Exception as e:
        logger.error(f"Error in streaming split: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Split text file into train/val/test by whole lines"
    )
    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help='Input text file to split'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Output directory (will create train/, val/, test/ subdirs)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training split ratio (default 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default 42)'
    )

    args = parser.parse_args()

    logger.info(f"Splitting {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Ratios: {args.train_ratio:.0%} train, {args.val_ratio:.0%} val, {1-args.train_ratio-args.val_ratio:.0%} test")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 70)

    result = split_file(
        args.input,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    if result:
        logger.info("=" * 70)
        logger.info(f"✓ Split complete: {result['total_lines']} lines processed")
        return 0
    else:
        logger.error("✗ Split failed")
        return 1


if __name__ == "__main__":
    exit(main())
