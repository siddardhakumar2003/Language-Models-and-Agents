#!/usr/bin/env python3
"""
Scrape Bhojpuri news sites, clean, deduplicate, and merge into corpus.

Targets Bhojpuri news portals and news sections with Devanagari content.
Implements proper rate limiting, state tracking, and deduplication.

Usage:
    python3 -m bhojpuri.data_collect.scrape_bhojpuri_news [--data-dir DATA_DIR] [--target-articles N]
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bhojpuri news sites with content in Devanagari
NEWS_SOURCES = [
    {
        "name": "Bihar Khabar (Bhojpuri)",
        "url": "https://www.bihartoday.com/",
        "selectors": {"articles": "article, .article, .post", "text": "p, .article-content"},
        "has_bhojpuri": True
    },
    {
        "name": "Bhojpuri Samachar",
        "url": "https://bhojpurisamachar.in/",
        "selectors": {"articles": "article, .post-item", "text": "p, .post-content"},
        "has_bhojpuri": True
    },
    {
        "name": "Local news aggregators (Hindi with Bhojpuri)",
        "url": "https://www.bhaskar.com/",
        "selectors": {"articles": ".article, .story", "text": "p, .story-content"},
        "has_bhojpuri": False  # May have some Bhojpuri content
    },
    {
        "name": "Aaj Tak (Hindi/Bhojpuri)",
        "url": "https://aajtak.intoday.in/",
        "selectors": {"articles": "article, .article-box", "text": "p"},
        "has_bhojpuri": False
    }
]


class BhojpuriNewsScraper:
    """Scrape Bhojpuri news sites with rate limiting and state tracking."""

    def __init__(self, data_dir: Path = None, state_file: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "news_raw"
        self.raw_dir.mkdir(exist_ok=True)

        if state_file is None:
            state_file = self.data_dir / "news_scrape_state.json"
        self.state_file = Path(state_file)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        self.state = self._load_state()
        self.batch_num = self.state.get("batch_num", 0)
        self.article_count = 0
        self.devanagari_ratio_threshold = 0.30

    def _load_state(self) -> Dict:
        """Load scraping state."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"urls_processed": {}, "batch_num": 0, "articles_extracted": 0}

    def _save_state(self):
        """Save scraping state."""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)

    def _is_devanagari_text(self, text: str) -> bool:
        """Check if text has sufficient Devanagari script."""
        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        total_chars = len([c for c in text if c.isalpha()])
        if total_chars == 0:
            return False
        ratio = devanagari_chars / total_chars
        return ratio >= self.devanagari_ratio_threshold

    def scrape_news_site(self, source: Dict, max_articles: int = 50) -> Generator[Dict, None, None]:
        """Scrape articles from a news source."""
        logger.info(f"Scraping {source['name']}...")
        articles_from_site = 0

        try:
            response = self.session.get(source['url'], timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article links
            article_selectors = source['selectors']['articles'].split(", ")
            articles = []
            for selector in article_selectors:
                articles.extend(soup.select(selector)[:max_articles])

            logger.info(f"  Found {len(articles)} potential article containers")

            for article in articles[:max_articles]:
                if articles_from_site >= max_articles:
                    break

                try:
                    # Try to extract link
                    link = None
                    link_elem = article.find('a', href=True)
                    if link_elem:
                        link = link_elem.get('href')
                        if link and not link.startswith('http'):
                            link = urljoin(source['url'], link)

                    # Extract text from article or fetch full article
                    text_selectors = source['selectors']['text'].split(", ")
                    text_parts = []
                    for selector in text_selectors:
                        for elem in article.select(selector):
                            p_text = elem.get_text(strip=True)
                            if p_text and len(p_text) > 20:
                                text_parts.append(p_text)

                    full_text = " ".join(text_parts)

                    # If article link available, try to fetch full content
                    if link and len(full_text) < 100:
                        try:
                            article_response = self.session.get(link, timeout=10)
                            article_soup = BeautifulSoup(article_response.content, 'html.parser')
                            for selector in text_selectors:
                                for elem in article_soup.select(selector):
                                    p_text = elem.get_text(strip=True)
                                    if p_text and len(p_text) > 20:
                                        text_parts.append(p_text)
                            full_text = " ".join(text_parts)
                            time.sleep(0.5)  # Rate limit
                        except Exception:
                            pass

                    # Check if text is Devanagari
                    if len(full_text) >= 100 and self._is_devanagari_text(full_text):
                        yield {
                            "text": full_text,
                            "source": source['name'],
                            "url": link or source['url']
                        }
                        articles_from_site += 1
                        self.article_count += 1

                except Exception as e:
                    logger.debug(f"Error processing article: {e}")
                    continue

                time.sleep(0.2)  # Rate limit between articles

        except Exception as e:
            logger.error(f"Error scraping {source['name']}: {e}")

        logger.info(f"  Extracted {articles_from_site} articles from {source['name']}")

    def run_scraping(self, target_articles: int = 500) -> Dict[str, int]:
        """Run news scraping across all sources."""
        logger.info("=" * 70)
        logger.info("BHOJPURI NEWS SCRAPING")
        logger.info("=" * 70)

        total_extracted = 0
        batch_file = None
        batch_texts = []
        batch_size = 100

        for source in NEWS_SOURCES:
            if total_extracted >= target_articles:
                break

            remaining = target_articles - total_extracted
            articles_per_source = max(10, remaining // len(NEWS_SOURCES))

            for article in self.scrape_news_site(source, max_articles=articles_per_source):
                batch_texts.append(article)
                total_extracted += 1

                if len(batch_texts) >= batch_size:
                    batch_file = self.raw_dir / f"news_batch_{self.batch_num:05d}.jsonl"
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        for article in batch_texts:
                            f.write(json.dumps({"text": article["text"]}, ensure_ascii=False) + "\n")
                    logger.info(f"Wrote batch {self.batch_num}: {batch_file.name}")
                    self.batch_num += 1
                    batch_texts = []

                if total_extracted >= target_articles:
                    break

        # Write final batch
        if batch_texts:
            batch_file = self.raw_dir / f"news_batch_{self.batch_num:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for article in batch_texts:
                    f.write(json.dumps({"text": article["text"]}, ensure_ascii=False) + "\n")
            logger.info(f"Wrote final batch {self.batch_num}: {batch_file.name}")

        self.state["articles_extracted"] = total_extracted
        self.state["batch_num"] = self.batch_num
        self._save_state()

        logger.info(f"Total articles scraped: {total_extracted}")
        return {"articles_scraped": total_extracted}


def process_news_scraping(data_dir: Path, target_articles: int = 500) -> Dict[str, int]:
    """Complete news scraping pipeline."""
    logger.info("=" * 70)
    logger.info("BHOJPURI NEWS SCRAPING PIPELINE")
    logger.info("=" * 70)

    # Scrape news
    scraper = BhojpuriNewsScraper(data_dir=data_dir)
    scrape_stats = scraper.run_scraping(target_articles=target_articles)

    raw_dir = data_dir / "news_raw"
    cleaned_dir = data_dir / "news_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    if not list(raw_dir.glob("*.jsonl")):
        logger.warning("No news articles scraped")
        return {"articles_scraped": 0}

    # Clean
    logger.info("Cleaning news texts...")
    cleaner = BhojpuriDataCleaner(
        input_dir=str(raw_dir),
        output_dir=str(cleaned_dir)
    )
    cleaner.run_cleaning()

    # Deduplicate
    logger.info("Deduplicating against existing corpus...")
    deduplicate_with_existing_corpus(
        data_dir=data_dir,
        source_cleaned_dir=cleaned_dir,
        output_cleaned_dir=cleaned_dir,
        state_file=None
    )

    # Merge
    logger.info("Merging into train/val/test splits...")
    stats = merge_ocr_into_splits(
        data_dir=data_dir,
        accumulator_filename="bhoj.txt",
        split_filename="bhoj.txt",
        cleaned_dir_name="news_cleaned",
        temp_batch_filename="news_new_batch.txt"
    )

    # Archive
    import shutil
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"news_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"News scraping complete: {stats}")
    logger.info("=" * 70)

    return {**stats, **scrape_stats}


def main():
    parser = argparse.ArgumentParser(description="Scrape Bhojpuri news sites")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory"
    )
    parser.add_argument(
        "--target-articles",
        type=int,
        default=500,
        help="Target number of articles (default: 500)"
    )
    args = parser.parse_args()

    process_news_scraping(args.data_dir, target_articles=args.target_articles)

    # Update config
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    news_source = "Bhojpuri News Scraping (Bihar Khabar, news sites, Devanagari filtered)"
    if news_source not in config.get("data_sources", []):
        config["data_sources"].append(news_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
