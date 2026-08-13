#!/usr/bin/env python3
"""
Update config.json with OCR data provenance and token progress tracking (Bhojpuri).

Computes total corpus tokens from actual file sizes, updates config with:
- ocr_txt_lines per split (appended to existing 'lines' field)
- token_progress block (total_corpus_tokens_estimate, target_tokens, progress_percent)
- Updates target_tokens to 500,000,000 if not already set

Usage:
    python3 update_config_ocr.py [--data-dir DATADIR]
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_corpus_tokens_estimate(data_dir: Path, language: str = "bhojpuri") -> Tuple[int, int]:
    """
    Compute total corpus tokens from actual file sizes.

    For Bhojpuri: bhoj.txt size // 4

    Returns (total_tokens, total_bytes)
    """
    total_bytes = 0

    if language == "bhojpuri":
        fpath = data_dir / "bhoj.txt"
        if fpath.exists():
            size = fpath.stat().st_size
            total_bytes += size
            logger.info(f"  bhoj.txt: {size / 1024 / 1024:.1f} MB")
    else:
        return 0, 0

    total_tokens = total_bytes // 4
    return total_tokens, total_bytes


def count_ocr_lines(data_dir: Path, cleaning_report_path: Path = None) -> Dict:
    """
    Count OCR-cleaned lines from cleaning_report.json if available,
    else sum cleaned_*.jsonl line counts directly.
    """
    result = {
        'total_kept': 0,
        'files_processed': 0,
        'pages_extracted': 0,
    }

    if cleaning_report_path and cleaning_report_path.exists():
        try:
            with open(cleaning_report_path, 'r') as f:
                report = json.load(f)
                result['total_kept'] = report.get('total_kept', 0)
                logger.info(f"Read cleaning_report.json: {result['total_kept']} lines kept")
                return result
        except Exception as e:
            logger.warning(f"Could not read cleaning_report.json: {e}")

    cleaned_dir = data_dir / "ocr_cleaned"
    if not cleaned_dir.exists():
        logger.warning(f"ocr_cleaned directory not found")
        return result

    for cleaned_jsonl in cleaned_dir.glob("cleaned_*.jsonl"):
        try:
            with open(cleaned_jsonl, 'r') as f:
                for line in f:
                    try:
                        json.loads(line)
                        result['total_kept'] += 1
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"Error reading {cleaned_jsonl}: {e}")

    logger.info(f"Counted {result['total_kept']} lines from ocr_cleaned/*.jsonl")
    return result


def get_ocr_state(data_dir: Path, state_file_name: str = "ocr_state.json") -> Dict:
    """Read ocr_state.json for file/page counts."""
    state_file = data_dir / state_file_name
    if not state_file.exists():
        return {'processed_files': {}, 'token_count': 0}

    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {state_file_name}: {e}")
        return {'processed_files': {}, 'token_count': 0}


def update_config_bhojpuri(config_path: Path, data_dir: Path) -> None:
    """Update Bhojpuri config.json with OCR data and token tracking."""
    logger.info("=" * 70)
    logger.info("UPDATING BHOJPURI CONFIG")
    logger.info("=" * 70)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get OCR state
    ocr_state = get_ocr_state(data_dir)
    ocr_info = count_ocr_lines(data_dir, data_dir / "ocr_cleaned" / "cleaning_report.json")

    logger.info(f"\nOCR Stats:")
    logger.info(f"  Processed files: {len(ocr_state.get('processed_files', {}))}")
    logger.info(f"  Total pages extracted: {sum(v.get('pages_extracted', 0) for v in ocr_state.get('processed_files', {}).values())}")
    logger.info(f"  Lines kept after cleaning: {ocr_info['total_kept']}")

    # Update OCR data block (Bhojpuri-specific: add ocr_txt_lines alongside existing 'lines')
    config.setdefault('ocr_data', {})
    config['ocr_data'] = {
        'source': 'OCR of scanned books/news images via ocr_sources/ (Tesseract, lang=hin, Devanagari proxy)',
        'files_processed': len(ocr_state.get('processed_files', {})),
        'pages_extracted': sum(v.get('pages_extracted', 0) for v in ocr_state.get('processed_files', {}).values()),
        'lines_kept_after_cleaning': ocr_info['total_kept'],
    }

    # Update target_tokens to 500M if not already 500M
    if config.get('target_tokens', 200_000_000) != 500_000_000:
        logger.info(f"Updating target_tokens from {config.get('target_tokens')} to 500,000,000")
        config['target_tokens'] = 500_000_000

    # Compute corpus token progress
    logger.info(f"\nComputing token progress...")
    total_tokens, total_bytes = compute_corpus_tokens_estimate(data_dir, "bhojpuri")
    target_tokens = config.get('target_tokens', 500_000_000)
    progress_percent = (total_tokens / target_tokens) * 100

    config.setdefault('token_progress', {})
    config['token_progress'] = {
        'total_corpus_tokens_estimate': total_tokens,
        'total_corpus_bytes': total_bytes,
        'target_tokens': target_tokens,
        'progress_percent': round(progress_percent, 2),
        'progress_str': f"{total_tokens / 1_000_000:.1f}M / {target_tokens / 1_000_000:.0f}M ({progress_percent:.1f}%)",
    }

    logger.info(f"  Total corpus: {config['token_progress']['progress_str']}")

    config['updated_at'] = datetime.now().isoformat()

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\n✓ Updated {config_path.name}")
    logger.info(f"  target_tokens: {config['target_tokens']}")
    logger.info(f"  token_progress: {config['token_progress']['progress_str']}")
    logger.info("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update config with OCR provenance and token progress (Bhojpuri)")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help='Data directory (default: bhojpuri/data/)'
    )

    args = parser.parse_args()
    config_path = args.data_dir / "config.json"

    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        return 1

    update_config_bhojpuri(config_path, args.data_dir)
    return 0


if __name__ == "__main__":
    exit(main())
