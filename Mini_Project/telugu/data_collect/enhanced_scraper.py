import os
import json
import time
import requests
from datetime import datetime
from typing import List, Generator, Set
from pathlib import Path
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import hashlib
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedTeluguScraper:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        self.seen_texts: Set[str] = set()
        self.token_count = 0
        self.target_tokens = 500_000_000
        self.scraped_urls: Set[str] = set()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)"""
        return max(1, len(text) // 4)

    def fetch_url(self, url: str, timeout: int = 15) -> str:
        """Fetch content from URL with retries"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response.text
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching {url}: {e}")
                return ""
        return ""

    def _is_telugu_text(self, text: str, min_length: int = 20) -> bool:
        """Check if text contains Telugu characters"""
        if not text or len(text) < min_length:
            return False

        telugu_unicode_start = 0x0C00
        telugu_unicode_end = 0x0C7F

        telugu_count = sum(1 for char in text if telugu_unicode_start <= ord(char) <= telugu_unicode_end)
        return telugu_count / len(text) > 0.25

    def scrape_wikipedia_telugu(self) -> Generator:
        """Scrape Telugu Wikipedia with limited scope, skipping TOC/infobox/references"""
        logger.info("Starting Wikipedia Telugu scraping...")

        # Key Telugu Wikipedia articles
        seed_urls = [
            "https://te.wikipedia.org/wiki/తెలుగు_భాష",
            "https://te.wikipedia.org/wiki/భారతదేశం",
            "https://te.wikipedia.org/wiki/ఆంధ్రప్రదేశ్",
            "https://te.wikipedia.org/wiki/తెలంగాణ",
            "https://te.wikipedia.org/wiki/చరిత్ర",
            "https://te.wikipedia.org/wiki/సంస్కృతి",
            "https://te.wikipedia.org/wiki/సాహిత్యం",
            "https://te.wikipedia.org/wiki/విజ్ఞానం",
        ]

        for url in seed_urls:
            if url in self.scraped_urls:
                continue

            self.scraped_urls.add(url)
            logger.info(f"Scraping: {url}")

            try:
                content = self.fetch_url(url)
                if not content:
                    continue

                soup = BeautifulSoup(content, 'html.parser')

                # Find main content div and skip TOC/infobox/references
                main_content = soup.find('div', id='mw-content-text')
                if not main_content:
                    main_content = soup

                # Remove unwanted sections before extraction
                for skip_elem in main_content.find_all(['div'], {'id': 'toc'}):
                    skip_elem.decompose()
                for skip_elem in main_content.find_all('table', class_='infobox'):
                    skip_elem.decompose()
                for skip_elem in main_content.find_all('sup', class_='reference'):
                    skip_elem.decompose()
                for skip_elem in main_content.find_all(['div', 'nav'], class_=lambda x: x and ('navbox' in x or 'navigation' in x)):
                    skip_elem.decompose()
                for skip_elem in main_content.find_all('span', class_='mw-editsection'):
                    skip_elem.decompose()

                # Extract only paragraphs from cleaned content
                for p in main_content.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and self._is_telugu_text(text):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(1)

            except Exception as e:
                logger.warning(f"Error scraping Wikipedia {url}: {e}")
                continue

    def scrape_wikibooks_telugu(self) -> Generator:
        """Scrape Telugu Wikibooks for educational content"""
        logger.info("Starting Wikibooks Telugu scraping...")

        # Wikibooks pages with direct URLs
        books = [
            "https://te.wikibooks.org/",
        ]

        for base_url in books:
            try:
                content = self.fetch_url(base_url)
                if not content:
                    continue

                soup = BeautifulSoup(content, 'html.parser')

                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and self._is_telugu_text(text, min_length=20):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(1)

            except Exception as e:
                logger.warning(f"Error scraping Wikibooks: {e}")

    def scrape_quotations(self) -> Generator:
        """Scrape Telugu quotations and proverbs"""
        logger.info("Scraping Telugu quotations...")

        try:
            # Wikiquote Telugu
            url = "https://te.wikiquote.org/wiki/ముఖ్య_పేజీ"

            content = self.fetch_url(url)
            if content:
                soup = BeautifulSoup(content, 'html.parser')

                for quote in soup.find_all(['p', 'blockquote', 'div']):
                    text = quote.get_text(strip=True)
                    if text and self._is_telugu_text(text, min_length=10):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(1)

        except Exception as e:
            logger.warning(f"Error scraping quotations: {e}")

    def scrape_wikisource_telugu(self) -> Generator:
        """Scrape Telugu Wikisource for literary content"""
        logger.info("Scraping Wikisource Telugu...")

        try:
            url = "https://te.wikisource.org/"

            content = self.fetch_url(url)
            if content:
                soup = BeautifulSoup(content, 'html.parser')

                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and self._is_telugu_text(text, min_length=20):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                for div in soup.find_all('div', class_=['mw-parser-output']):
                    for p in div.find_all('p'):
                        text = p.get_text(strip=True)
                        if text and self._is_telugu_text(text, min_length=20):
                            text_hash = hashlib.md5(text.encode()).hexdigest()
                            if text_hash not in self.seen_texts:
                                self.seen_texts.add(text_hash)
                                yield text

                time.sleep(1)

        except Exception as e:
            logger.warning(f"Error scraping Wikisource: {e}")

    def scrape_news_sites(self) -> Generator:
        """Scrape Telugu news websites, avoiding nav/menu/footer chrome"""
        logger.info("Scraping news sites...")

        news_sources = [
            "https://www.eenadu.net/",
            "https://www.greatandhra.com/",
        ]

        skip_patterns = ['nav', 'menu', 'footer', 'aside', 'widget', 'related', 'most-read', 'sidebar']

        for url in news_sources:
            try:
                content = self.fetch_url(url)
                if not content:
                    continue

                soup = BeautifulSoup(content, 'html.parser')

                # Remove nav/menu/footer/widget elements before extraction
                for elem in soup.find_all(['nav', 'footer', 'aside']):
                    elem.decompose()

                for elem in soup.find_all(['div', 'section']):
                    class_str = (elem.get('class', []) or []) if elem.get('class') else []
                    id_str = (elem.get('id') or '').lower()
                    class_lower = ' '.join(class_str).lower() if class_str else ''

                    if any(pat in class_lower or pat in id_str for pat in skip_patterns):
                        elem.decompose()

                # Extract only paragraphs and article tags
                for elem in soup.find_all(['p', 'article']):
                    text = elem.get_text(strip=True)
                    if text and self._is_telugu_text(text, min_length=20):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(2)

            except Exception as e:
                logger.warning(f"Error scraping news site {url}: {e}")

    def save_batch(self, texts: List[str], batch_id: int) -> int:
        """Save batch and return token count"""
        if not texts:
            return 0

        output_file = self.raw_dir / f"batch_{batch_id:06d}.jsonl"

        total_tokens = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                tokens = self.estimate_tokens(text)
                total_tokens += tokens
                json.dump({
                    'text': text,
                    'tokens': tokens,
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False)
                f.write('\n')

        return total_tokens

    def run_scraper_continuous(self, max_tokens: int = 100_000_000, batch_size: int = 50) -> None:
        """Run continuous scraper until reaching token target"""
        logger.info(f"Starting scraper. Target: {max_tokens / 1e6:.1f}M tokens")

        current_batch = []
        batch_id = 0
        self.token_count = 0

        start_time = time.time()

        # Combine multiple sources
        sources = [
            ("Wikipedia", self.scrape_wikipedia_telugu()),
            ("News Sites", self.scrape_news_sites()),
            ("Wikibooks", self.scrape_wikibooks_telugu()),
            ("Wikiquote", self.scrape_quotations()),
            ("Wikisource", self.scrape_wikisource_telugu()),
        ]

        for source_name, source_generator in sources:
            logger.info(f"Processing source: {source_name}")

            try:
                for text in source_generator:
                    if not text or len(text) < 20:
                        continue

                    current_batch.append(text)

                    if len(current_batch) >= batch_size:
                        batch_tokens = self.save_batch(current_batch, batch_id)
                        self.token_count += batch_tokens
                        batch_id += 1

                        progress_pct = (self.token_count / max_tokens) * 100
                        logger.info(
                            f"Batch {batch_id-1}: {batch_tokens/1e6:.3f}M tokens | "
                            f"Total: {self.token_count/1e6:.2f}M / {max_tokens/1e6:.1f}M ({progress_pct:.1f}%) "
                            f"[{source_name}]"
                        )

                        current_batch = []

                        if self.token_count >= max_tokens:
                            logger.info("Target tokens reached!")
                            break

                if self.token_count >= max_tokens:
                    break

            except Exception as e:
                logger.error(f"Error in source {source_name}: {e}")
                continue

        # Save remaining batch
        if current_batch:
            batch_tokens = self.save_batch(current_batch, batch_id)
            self.token_count += batch_tokens

        elapsed_time = time.time() - start_time

        # Final statistics
        if self.raw_dir.exists():
            total_size_mb = sum(f.stat().st_size for f in self.raw_dir.glob('*.jsonl')) / 1024 / 1024
            total_files = len(list(self.raw_dir.glob('*.jsonl')))
        else:
            total_size_mb = 0
            total_files = 0

        logger.info(f"\n{'='*70}")
        logger.info(f"Scraping completed")
        logger.info(f"Time elapsed: {elapsed_time / 60:.2f} minutes")
        logger.info(f"Total tokens collected: {self.token_count / 1e6:.2f}M")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total size: {total_size_mb:.2f} MB")
        if elapsed_time > 0:
            logger.info(f"Average tokens per second: {self.token_count / elapsed_time:.0f}")
        logger.info(f"Data directory: {self.output_dir.absolute()}")
        logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    scraper = EnhancedTeluguScraper()
    scraper.run_scraper_continuous(max_tokens=10_000_000, batch_size=50)
