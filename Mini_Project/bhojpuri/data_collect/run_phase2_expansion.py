#!/usr/bin/env python3
"""
Master orchestration script: Run all Phase 2 Bhojpuri corpus expansion sources sequentially.

This script runs all 6 data sources in order:
1. Wikipedia dump (authentic Bhojpuri Wikipedia)
2. fossdot/bhojpuri-corpus (GitHub community corpus)
3. BHLTR (monolingual + POS-annotated)
4. FineWeb-2 (CommonCrawl web text)
5. archive.org (Bhojpuri-tagged books, pre-extracted djvu.txt)
6. Manual Tesseract OCR (if PDFs are available in ocr_sources/)

After all sources are merged, generates a final report showing:
- Total corpus size before expansion
- Total corpus size after expansion
- Per-source contribution (lines/tokens)
- Progress toward 500M token target

Usage:
    python3 -m bhojpuri.data_collect.run_phase2_expansion [OPTIONS]

Options:
    --data-dir DIR              Data directory (default: ../data)
    --skip-wiki                 Skip Wikipedia dump
    --skip-fossdot              Skip fossdot corpus
    --skip-bhltr                Skip BHLTR corpus
    --skip-fineweb2             Skip FineWeb-2 corpus
    --skip-archive              Skip archive.org Bhojpuri books
    --skip-ocr                  Skip manual OCR (if PDFs available)
    --archive-target-items N    Target number of archive.org items (default: 500)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Import all source processing functions
try:
    from .download_wikipedia_dump import process_wiki_dump
    from .download_github_corpus import process_github_corpus
    from .download_bhltr_corpus import process_bhltr_corpus
    from .download_fineweb2_corpus import process_fineweb2_corpus
    from .download_archive_org_bhojpuri import process_archive_org_bhojpuri
    from .update_config_ocr_fixed import update_config_bhojpuri

    # OCR is optional
    try:
        from .ocr_books_manual import process_manual_ocr
        ocr_available = True
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"OCR module not available: {e}")
        logger.warning("  (Tesseract/PyMuPDF not installed; manual OCR will be skipped)")
        process_manual_ocr = None
        ocr_available = False

except ImportError as e:
    print(f"Error importing submodules: {e}")
    print("Make sure you're running this as a module: python3 -m bhojpuri.data_collect.run_phase2_expansion")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_corpus_stats(data_dir: Path) -> Tuple[int, int, int]:
    """Read current corpus stats from config.json."""
    config_path = data_dir / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        train_lines = config.get('splits', {}).get('train', {}).get('lines', 0)
        tokens = config.get('token_progress', {}).get('total_corpus_tokens_estimate', 0)
        total_lines = config.get('total_training_lines', 0)
        return train_lines, total_lines, tokens
    except Exception as e:
        logger.error(f"Failed to read corpus stats: {e}")
        return 0, 0, 0


def run_phase2_expansion(
    data_dir: Path,
    skip_wiki: bool = False,
    skip_fossdot: bool = False,
    skip_bhltr: bool = False,
    skip_fineweb2: bool = False,
    skip_archive: bool = False,
    skip_ocr: bool = False,
    archive_target_items: int = 500
) -> Dict:
    """Run all Phase 2 corpus expansion sources."""

    logger.info("=" * 80)
    logger.info("PHASE 2: BHOJPURI CORPUS EXPANSION (All Sources)")
    logger.info("=" * 80)

    # Record pre-expansion stats
    logger.info("Recording pre-expansion corpus statistics...")
    pre_train_lines, pre_total_lines, pre_tokens = read_corpus_stats(data_dir)
    logger.info(f"Pre-expansion: {pre_total_lines:,} total lines / ~{pre_tokens/1e6:.1f}M tokens")

    results = {}
    failed_sources = []

    # Step 1: Wikipedia Dump
    if not skip_wiki:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1/6: BHOJPURI WIKIPEDIA DUMP")
        logger.info("=" * 80)
        try:
            stats = process_wiki_dump(data_dir)
            results['wikipedia'] = stats
            logger.info(f"✓ Wikipedia processing complete: {stats}")
        except Exception as e:
            logger.error(f"✗ Wikipedia processing failed: {e}")
            failed_sources.append(('Wikipedia', str(e)))

    # Step 2: fossdot/bhojpuri-corpus
    if not skip_fossdot:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2/6: FOSSDOT/BHOJPURI-CORPUS")
        logger.info("=" * 80)
        try:
            stats = process_github_corpus(data_dir)
            results['fossdot'] = stats
            logger.info(f"✓ fossdot processing complete: {stats}")
        except Exception as e:
            logger.error(f"✗ fossdot processing failed: {e}")
            failed_sources.append(('fossdot', str(e)))

    # Step 3: BHLTR
    if not skip_bhltr:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3/6: BHLTR CORPUS")
        logger.info("=" * 80)
        try:
            stats = process_bhltr_corpus(data_dir)
            results['bhltr'] = stats
            logger.info(f"✓ BHLTR processing complete: {stats}")
        except Exception as e:
            logger.error(f"✗ BHLTR processing failed: {e}")
            failed_sources.append(('BHLTR', str(e)))

    # Step 4: FineWeb-2
    if not skip_fineweb2:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4/6: FINEWEB-2 BHO_DEVA CORPUS")
        logger.info("=" * 80)
        try:
            stats = process_fineweb2_corpus(data_dir)
            results['fineweb2'] = stats
            logger.info(f"✓ FineWeb-2 processing complete: {stats}")
        except Exception as e:
            logger.error(f"✗ FineWeb-2 processing failed: {e}")
            failed_sources.append(('FineWeb-2', str(e)))

    # Step 5: archive.org Bhojpuri Books
    if not skip_archive:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5/6: ARCHIVE.ORG BHOJPURI BOOKS (Pre-extracted djvu.txt)")
        logger.info("=" * 80)
        try:
            stats = process_archive_org_bhojpuri(data_dir, target_items=archive_target_items)
            results['archive_org'] = stats
            logger.info(f"✓ archive.org processing complete: {stats}")
        except Exception as e:
            logger.error(f"✗ archive.org processing failed: {e}")
            failed_sources.append(('archive.org', str(e)))

    # Step 6: Manual Tesseract OCR (if PDFs available and OCR module is available)
    if not skip_ocr:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6/6: MANUAL TESSERACT OCR (if PDFs in ocr_sources/)")
        logger.info("=" * 80)
        if process_manual_ocr is None:
            logger.info("⊘ OCR module not available (pytesseract/PyMuPDF not installed)")
            logger.info("  To use OCR: pip install pytesseract PyMuPDF")
            results['manual_ocr'] = {"pdfs_processed": 0, "texts_extracted": 0, "skipped": "OCR not available"}
        else:
            try:
                stats = process_manual_ocr(data_dir)
                results['manual_ocr'] = stats
                if stats.get('pdfs_processed', 0) > 0:
                    logger.info(f"✓ Manual OCR processing complete: {stats}")
                else:
                    logger.info("No PDFs found in ocr_sources/; skipping OCR. (Manual curation required)")
            except Exception as e:
                logger.error(f"✗ Manual OCR processing failed: {e}")
                failed_sources.append(('Manual OCR', str(e)))

    # Final config update
    logger.info("\n" + "=" * 80)
    logger.info("FINAL: UPDATE CONFIG & GENERATE REPORT")
    logger.info("=" * 80)
    try:
        config_path = data_dir / "config.json"
        update_config_bhojpuri(config_path, data_dir)
        logger.info("✓ config.json updated with final statistics")
    except Exception as e:
        logger.error(f"✗ Config update failed: {e}")
        failed_sources.append(('Config Update', str(e)))

    # Read post-expansion stats
    post_train_lines, post_total_lines, post_tokens = read_corpus_stats(data_dir)

    # Generate final report
    logger.info("\n" + "=" * 80)
    logger.info("FINAL REPORT")
    logger.info("=" * 80)

    report = {
        "timestamp": datetime.now().isoformat(),
        "pre_expansion": {
            "total_lines": pre_total_lines,
            "tokens": pre_tokens,
        },
        "post_expansion": {
            "total_lines": post_total_lines,
            "tokens": post_tokens,
        },
        "expansion_delta": {
            "lines_added": post_total_lines - pre_total_lines,
            "tokens_added": post_tokens - pre_tokens,
        },
        "source_results": results,
        "failed_sources": failed_sources,
    }

    # Print formatted report
    logger.info(f"\n{'Corpus Size Summary':^80}")
    logger.info("-" * 80)
    logger.info(f"Pre-expansion:  {pre_total_lines:>15,} lines / {pre_tokens/1e6:>7.1f}M tokens")
    logger.info(f"Post-expansion: {post_total_lines:>15,} lines / {post_tokens/1e6:>7.1f}M tokens")
    logger.info(f"Added:          {report['expansion_delta']['lines_added']:>15,} lines / {report['expansion_delta']['tokens_added']/1e6:>7.1f}M tokens")
    logger.info("-" * 80)

    # Calculate progress to target
    target_tokens = 500_000_000
    progress_percent = (post_tokens / target_tokens) * 100
    logger.info(f"Progress to 500M target: {post_tokens/1e6:>7.1f}M / 500.0M ({progress_percent:>6.2f}%)")
    logger.info("")

    logger.info(f"{'Per-Source Contribution':^80}")
    logger.info("-" * 80)
    for source_name, source_stats in results.items():
        if isinstance(source_stats, dict):
            lines = source_stats.get('lines_appended', 0)
            logger.info(f"{source_name:30s}: {lines:>10,} lines appended")
    logger.info("-" * 80)

    if failed_sources:
        logger.warning(f"\n{'FAILURES':^80}")
        logger.warning("-" * 80)
        for source_name, error in failed_sources:
            logger.warning(f"{source_name:30s}: {error}")
        logger.warning("-" * 80)

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 EXPANSION COMPLETE")
    logger.info("=" * 80)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run all Phase 2 Bhojpuri corpus expansion sources sequentially"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    parser.add_argument("--skip-wiki", action="store_true", help="Skip Wikipedia dump")
    parser.add_argument("--skip-fossdot", action="store_true", help="Skip fossdot corpus")
    parser.add_argument("--skip-bhltr", action="store_true", help="Skip BHLTR corpus")
    parser.add_argument("--skip-fineweb2", action="store_true", help="Skip FineWeb-2 corpus")
    parser.add_argument("--skip-archive", action="store_true", help="Skip archive.org Bhojpuri books")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip manual Tesseract OCR")
    parser.add_argument(
        "--archive-target-items",
        type=int,
        default=500,
        help="Target number of archive.org items (default: 500)"
    )

    args = parser.parse_args()

    report = run_phase2_expansion(
        args.data_dir,
        skip_wiki=args.skip_wiki,
        skip_fossdot=args.skip_fossdot,
        skip_bhltr=args.skip_bhltr,
        skip_fineweb2=args.skip_fineweb2,
        skip_archive=args.skip_archive,
        skip_ocr=args.skip_ocr,
        archive_target_items=args.archive_target_items
    )

    # Save report to file
    report_file = args.data_dir / "phase2_expansion_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
