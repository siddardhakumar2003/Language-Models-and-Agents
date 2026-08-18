#!/usr/bin/env python3
"""
Download and parse Bhojpuri Wikipedia dump, clean, deduplicate, and merge into splits.

Downloads bhwiki-latest-pages-articles.xml.bz2, parses to plain text,
runs through BhojpuriDataCleaner, deduplicates against existing corpus,
and appends to train/val/test splits.

Usage:
    python3 -m bhojpuri.data_collect.download_wikipedia_dump [--data-dir DATA_DIR]
"""

import argparse
import bz2
import json
import logging
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator
from xml.etree import ElementTree as ET

from tqdm import tqdm

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WIKI_DUMP_URL = "https://dumps.wikimedia.org/bhwiki/latest/bhwiki-latest-pages-articles.xml.bz2"


def download_wikipedia_dump(data_dir: Path) -> Path:
    """Download the Bhojpuri Wikipedia dump."""
    dump_dir = data_dir / "wiki_raw_download"
    dump_dir.mkdir(exist_ok=True)

    dump_file = dump_dir / "bhwiki-latest-pages-articles.xml.bz2"

    if dump_file.exists():
        logger.info(f"Wikipedia dump already exists at {dump_file}")
        return dump_file

    logger.info(f"Downloading Bhojpuri Wikipedia dump from {WIKI_DUMP_URL}")
    result = subprocess.run(
        ["wget", "-c", "-O", str(dump_file), WIKI_DUMP_URL],
        capture_output=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download Wikipedia dump: {result.returncode}")

    logger.info(f"Downloaded Wikipedia dump: {dump_file.stat().st_size / 1024 / 1024:.1f} MB")
    return dump_file


def strip_wikitext(text: str) -> str:
    """Simple wikitext stripping without mwparserfromhell."""
    # Remove [[links]]
    text = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', text)
    # Remove {{templates}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Remove <tags>
    text = re.sub(r'<[^>]+>', '', text)
    # Remove ---- (horizontal rules)
    text = re.sub(r'-{4,}', '', text)
    # Remove == headings ==
    text = re.sub(r'={2,}(.+?)={2,}', r'\1', text)
    # Remove ' '' ''' bold/italic markers
    text = re.sub(r"'{2,}", '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_articles_from_dump(dump_file: Path) -> Generator[Dict[str, str], None, None]:
    """Extract article text from Wikipedia XML dump using ElementTree."""
    logger.info("Parsing Wikipedia dump...")

    namespaces = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
    article_count = 0

    with bz2.open(dump_file, 'rt', encoding='utf-8', errors='replace') as f:
        try:
            tree = ET.parse(f)
            root = tree.getroot()

            for page in root.findall('.//mw:page', namespaces):
                article_count += 1
                if article_count % 1000 == 0:
                    logger.info(f"  Processed {article_count} articles...")

                # Get title
                title_elem = page.find('mw:title', namespaces)
                if title_elem is None:
                    continue
                title = title_elem.text or ""

                # Skip redirect pages
                redirect = page.find('mw:redirect', namespaces)
                if redirect is not None:
                    continue

                # Get namespace (0 = main article namespace)
                ns_elem = page.find('mw:ns', namespaces)
                if ns_elem is not None and ns_elem.text and ns_elem.text != '0':
                    continue

                # Get latest revision text
                revision = page.find('.//mw:revision[last()]', namespaces)
                if revision is None:
                    continue

                text_elem = revision.find('mw:text', namespaces)
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text
                if not text or len(text.strip()) < 50:
                    continue

                # Strip wikitext markup
                plain_text = strip_wikitext(text)

                if len(plain_text) >= 50:
                    yield {
                        "text": plain_text,
                        "title": title
                    }
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            logger.info("Retrying with incremental parsing...")
            # Fallback: try incremental parsing for large files
            f.seek(0)
            article_count = 0
            for event, elem in ET.iterparse(f, events=['end']):
                if elem.tag.endswith('}page'):
                    article_count += 1
                    if article_count % 1000 == 0:
                        logger.info(f"  Processed {article_count} articles...")

                    title_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title')
                    if title_elem is None or not title_elem.text:
                        elem.clear()
                        continue

                    redirect = elem.find('{http://www.mediawiki.org/xml/export-0.10/}redirect')
                    if redirect is not None:
                        elem.clear()
                        continue

                    ns_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}ns')
                    if ns_elem is not None and ns_elem.text and ns_elem.text != '0':
                        elem.clear()
                        continue

                    revision = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}revision')
                    if revision is None:
                        elem.clear()
                        continue

                    text_elem = revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
                    if text_elem is None or text_elem.text is None:
                        elem.clear()
                        continue

                    text = text_elem.text
                    if not text or len(text.strip()) < 50:
                        elem.clear()
                        continue

                    plain_text = strip_wikitext(text)
                    if len(plain_text) >= 50:
                        yield {"text": plain_text, "title": title_elem.text}

                    elem.clear()


def process_wiki_dump(data_dir: Path) -> Dict[str, int]:
    """Download, parse, clean, deduplicate, and merge Wikipedia dump."""
    logger.info("=" * 70)
    logger.info("WIKIPEDIA DUMP PROCESSING")
    logger.info("=" * 70)

    # Download dump
    dump_file = download_wikipedia_dump(data_dir)

    # Create raw and cleaned directories
    raw_dir = data_dir / "wiki_raw"
    cleaned_dir = data_dir / "wiki_cleaned"
    raw_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    # Extract and batch articles
    logger.info("Extracting and batching articles...")
    batch_size = 1000
    batch_num = 0
    batch_texts = []
    article_count = 0

    for article in extract_articles_from_dump(dump_file):
        batch_texts.append(article)
        article_count += 1

        if len(batch_texts) >= batch_size:
            # Write batch to JSONL
            batch_file = raw_dir / f"wiki_batch_{batch_num:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for text in batch_texts:
                    f.write(json.dumps({"text": text["text"]}, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {batch_size} articles to {batch_file.name}")
            batch_num += 1
            batch_texts = []

    # Write final partial batch
    if batch_texts:
        batch_file = raw_dir / f"wiki_batch_{batch_num:05d}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as f:
            for text in batch_texts:
                f.write(json.dumps({"text": text["text"]}, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(batch_texts)} articles to {batch_file.name}")

    logger.info(f"Total articles extracted: {article_count}")

    # Clean articles
    logger.info("Cleaning Wikipedia articles...")
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
        cleaned_dir_name="wiki_cleaned",
        temp_batch_filename="wiki_new_batch.txt"
    )

    # Clean up downloads
    logger.info("Cleaning up intermediate files...")
    if (data_dir / "wiki_raw_download").exists():
        shutil.rmtree(data_dir / "wiki_raw_download")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Archive cleaned dir
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"wiki_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived cleaned data to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"Wikipedia processing complete: {stats}")
    logger.info("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and process Bhojpuri Wikipedia dump")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: ../data)"
    )
    args = parser.parse_args()

    process_wiki_dump(args.data_dir)

    # Update config
    logger.info("Updating config.json...")
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Append source to list if not already there
    wiki_source = "bho.wikipedia.org (bhwiki-latest-pages-articles dump, ~8,857 articles)"
    if wiki_source not in config.get("data_sources", []):
        config["data_sources"].append(wiki_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
