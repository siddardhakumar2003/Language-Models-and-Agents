#!/usr/bin/env python3
"""
Download and extract Hindi Wikipedia dump for translation to Bhojpuri.

Downloads hiwiki-latest-pages-articles.xml.bz2 (~239MB), parses to plain text batches,
writes raw-text JSONL to hindi_wiki_raw (no cleaning/dedup/merge here —
Phase 3's translator pipeline handles that for Hindi→Bhojpuri MT).

Usage:
    python3 -m bhojpuri.data_collect.download_hindi_wikipedia_dump [--data-dir DATA_DIR] [--max-articles N]
"""

import argparse
import bz2
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Generator, Optional

from .download_wikipedia_dump import strip_wikitext, extract_articles_from_dump

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WIKI_DUMP_URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"


def download_hindi_wikipedia_dump(data_dir: Path) -> Path:
    """Download the Hindi Wikipedia dump."""
    dump_dir = data_dir / "hindi_wiki_raw_download"
    dump_dir.mkdir(exist_ok=True, parents=True)

    dump_file = dump_dir / "hiwiki-latest-pages-articles.xml.bz2"

    if dump_file.exists():
        logger.info(f"Hindi Wikipedia dump already exists at {dump_file} ({dump_file.stat().st_size / 1024 / 1024:.1f} MB)")
        return dump_file

    logger.info(f"Downloading Hindi Wikipedia dump from {WIKI_DUMP_URL}")
    logger.info("(This is ~239MB; may take a few minutes depending on bandwidth)")
    result = subprocess.run(
        ["wget", "-c", "-O", str(dump_file), WIKI_DUMP_URL],
        capture_output=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download Hindi Wikipedia dump: {result.returncode}")

    logger.info(f"Downloaded Hindi Wikipedia dump: {dump_file.stat().st_size / 1024 / 1024:.1f} MB")
    return dump_file


def process_hindi_wiki_dump(data_dir: Path, max_articles: Optional[int] = None) -> Dict[str, int]:
    """Download, parse, and extract Hindi Wikipedia dump to raw-text batches (no cleaning/dedup/merge)."""
    logger.info("=" * 70)
    logger.info("HINDI WIKIPEDIA DUMP PROCESSING (for translation)")
    logger.info("=" * 70)

    # Download dump
    dump_file = download_hindi_wikipedia_dump(data_dir)

    # Create raw directory (cleaned/merge stages happen in Phase 3's translator pipeline)
    raw_dir = data_dir / "hindi_wiki_raw"
    raw_dir.mkdir(exist_ok=True, parents=True)

    # Extract and batch articles
    logger.info(f"Extracting articles from Hindi Wikipedia dump{' (max: ' + str(max_articles) + ')' if max_articles else ''}...")
    batch_size = 1000
    batch_num = 0
    batch_texts = []
    article_count = 0

    for article in extract_articles_from_dump(dump_file):
        batch_texts.append(article)
        article_count += 1

        if max_articles and article_count >= max_articles:
            logger.info(f"Reached max_articles limit ({max_articles}), stopping extraction")
            break

        if len(batch_texts) >= batch_size:
            # Write batch to JSONL (schema: {"text": ...} only, matching hindi_news_raw/hindi_books_raw)
            batch_file = raw_dir / f"hindi_wiki_batch_{batch_num:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for text in batch_texts:
                    f.write(json.dumps({"text": text["text"]}, ensure_ascii=False) + "\n")
            logger.info(f"  Wrote batch {batch_num}: {batch_size} articles to {batch_file.name}")
            batch_num += 1
            batch_texts = []

    # Write final partial batch
    if batch_texts:
        batch_file = raw_dir / f"hindi_wiki_batch_{batch_num:05d}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as f:
            for text in batch_texts:
                f.write(json.dumps({"text": text["text"]}, ensure_ascii=False) + "\n")
        logger.info(f"  Wrote batch {batch_num}: {len(batch_texts)} articles to {batch_file.name}")

    logger.info(f"Total articles extracted: {article_count}")

    # Delete the downloaded dump to save space (it's not needed after extraction)
    logger.info("Cleaning up downloaded .bz2 file...")
    if dump_file.exists():
        dump_file.unlink()
        logger.info(f"Deleted {dump_file.name}")

    logger.info("=" * 70)
    logger.info(f"Hindi Wikipedia extraction complete: {article_count} articles → {batch_num + 1} batches in hindi_wiki_raw/")
    logger.info("Next step: Phase 3 translator will read from hindi_wiki_raw alongside hindi_news_raw and hindi_books_raw")
    logger.info("=" * 70)

    return {
        "articles_extracted": article_count,
        "batches_written": batch_num + 1 if batch_texts else batch_num,
    }


def main():
    parser = argparse.ArgumentParser(description="Download and extract Hindi Wikipedia dump for translation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Limit extraction to N articles (default: None = full dump, ~160K articles)"
    )
    args = parser.parse_args()

    result = process_hindi_wiki_dump(args.data_dir, max_articles=args.max_articles)
    logger.info(f"Extraction result: {result}")


if __name__ == "__main__":
    main()
