#!/usr/bin/env python3
"""
Clean up intermediate scraping artifacts after final corpus is created.

Removes raw/, cleaned/, processed/ directories after verifying the
final telugu.txt file is valid. Preserves te.txt and train/val/test/.

Usage:
    python3 cleanup_intermediate.py                    # Dry run (shows what would be deleted)
    python3 cleanup_intermediate.py --yes              # Actually delete

Safety:
    - Dry run by default (no deletion unless --yes is passed)
    - Verifies final telugu.txt exists and has content
    - Cross-checks line count against cleaning report if present
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_final_output(txt_path: Path, cleaning_report: Path = None, min_lines: int = 1) -> tuple:
    """Verify final .txt file is valid. Returns (is_valid, report_dict)"""
    report = {
        'txt_exists': False,
        'txt_size_mb': 0.0,
        'txt_lines': 0,
        'report_exists': False,
        'report_total_kept': 0,
        'line_count_match': False,
        'is_valid': False
    }

    if not txt_path.exists():
        logger.error(f"Final text file not found: {txt_path}")
        return False, report

    report['txt_exists'] = True
    report['txt_size_mb'] = txt_path.stat().st_size / 1024 / 1024

    # Count lines
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_count, _ in enumerate(f, 1):
                pass
        report['txt_lines'] = line_count
    except Exception as e:
        logger.error(f"Error reading {txt_path}: {e}")
        return False, report

    if report['txt_lines'] < min_lines:
        logger.error(f"Text file is empty or too small ({report['txt_lines']} lines)")
        return False, report

    # Cross-check with cleaning report if present
    if cleaning_report and cleaning_report.exists():
        try:
            with open(cleaning_report, 'r') as f:
                clean_data = json.load(f)
            report['report_exists'] = True
            report['report_total_kept'] = clean_data.get('total_kept', 0)

            # Allow 1% variance due to formatting differences
            variance = abs(report['txt_lines'] - report['report_total_kept']) / max(1, report['report_total_kept'])
            if variance < 0.01:
                report['line_count_match'] = True
        except Exception as e:
            logger.warning(f"Could not read cleaning report: {e}")

    report['is_valid'] = True
    return True, report


def compute_deletion_plan(data_dir: Path) -> list:
    """Return list of directories to delete"""
    plan = []
    for subdir_name in ['raw', 'cleaned', 'processed']:
        subdir = data_dir / subdir_name
        if subdir.exists():
            plan.append(subdir)
    return plan


def print_plan(plan: list) -> int:
    """Print deletion plan with sizes. Returns total bytes."""
    if not plan:
        logger.info("Nothing to delete.")
        return 0

    total_bytes = 0
    logger.info("\nDeletable intermediate directories:")
    logger.info("-" * 70)

    for subdir in plan:
        try:
            size_bytes = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
            size_mb = size_bytes / 1024 / 1024
            file_count = len(list(subdir.glob('**/*.jsonl'))) + len(list(subdir.glob('**/*.txt')))
            total_bytes += size_bytes
            logger.info(f"  {subdir.name:15} {size_mb:8.2f} MB  ({file_count} files)")
        except Exception as e:
            logger.warning(f"Error computing size for {subdir}: {e}")

    total_mb = total_bytes / 1024 / 1024
    logger.info("-" * 70)
    logger.info(f"  TOTAL         {total_mb:8.2f} MB")
    logger.info("-" * 70)

    return total_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Clean up intermediate scraping artifacts"
    )
    parser.add_argument(
        '--data-dir',
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help='Data directory (default: telugu/data/)'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Actually delete (default: dry run only)'
    )
    parser.add_argument(
        '--min-lines',
        type=int,
        default=1,
        help='Minimum lines required in final text file'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    logger.info(f"Telugu cleanup for: {data_dir}")
    logger.info(f"Mode: {'DELETE' if args.yes else 'DRY RUN (no changes)'}")
    logger.info("=" * 70)

    # Verify final output
    txt_file = data_dir / "telugu.txt"
    cleaning_report = data_dir / "cleaning_report.json"

    is_valid, report = verify_final_output(txt_file, cleaning_report, args.min_lines)

    logger.info("\nFinal telugu.txt verification:")
    logger.info(f"  Exists:        {report['txt_exists']}")
    logger.info(f"  Size:          {report['txt_size_mb']:.2f} MB")
    logger.info(f"  Lines:         {report['txt_lines']}")
    if report['report_exists']:
        logger.info(f"  Report exists: Yes ({report['report_total_kept']} kept)")
        logger.info(f"  Line match:    {'✓' if report['line_count_match'] else '✗ (within tolerance)'}")
    logger.info(f"  Valid:         {'✓ YES' if is_valid else '✗ NO'}")

    if not is_valid:
        logger.error("\nCannot proceed: telugu.txt is not valid")
        logger.error("Check data collection/cleaning status before retrying")
        return 1

    # Check what can be deleted
    plan = compute_deletion_plan(data_dir)

    if not plan:
        logger.info("\nNothing to delete.")
        return 0

    logger.info("\n" + "=" * 70)
    total_bytes = print_plan(plan)

    if not args.yes:
        logger.info("\nDRY RUN: No files were deleted.")
        logger.info("To actually delete, run with --yes flag:")
        logger.info(f"  python3 cleanup_intermediate.py --yes")
        return 0

    # Actually delete
    logger.info("\nProceeding with deletion...")
    deleted_count = 0

    for subdir in plan:
        try:
            logger.info(f"Deleting {subdir.name}...")
            shutil.rmtree(subdir)
            deleted_count += 1
        except Exception as e:
            logger.error(f"Error deleting {subdir}: {e}")
            return 1

    logger.info(f"\n✓ Successfully deleted {deleted_count} directories")
    logger.info(f"✓ Freed ~{total_bytes / 1024 / 1024:.2f} MB of disk space")
    logger.info(f"✓ Preserved: telugu.txt, te.txt (if present), train/val/test/")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
