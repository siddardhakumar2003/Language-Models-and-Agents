import os
import json
import time
import requests
from datetime import datetime
from typing import List, Generator
from pathlib import Path
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BhojpuriTextScraper:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.seen_urls = set()
        self.seen_texts = set()  # For deduplication
        self.stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'total_paragraphs': 0
        }

        # Bhojpuri Unicode range (Devanagari script)
        self.bhojpuri_unicode_start = 0x0900
        self.bhojpuri_unicode_end = 0x097F

    def fetch_url(self, url: str, timeout: int = 10) -> str:
        """Fetch content from URL with error handling"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.encoding = 'utf-8'
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""

    def extract_bhojpuri_text_wikipedia(self, url: str) -> List[str]:
        """Extract Bhojpuri/Devanagari text from Wikipedia pages"""
        content = self.fetch_url(url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 50 and self._is_devanagari_text(text):
                paragraphs.append(text)

        return paragraphs

    def extract_bhojpuri_text_generic(self, url: str) -> List[str]:
        """Extract Devanagari text from generic websites"""
        content = self.fetch_url(url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')

        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        paragraphs = []
        for p in soup.find_all(['p', 'div', 'article']):
            text = p.get_text(strip=True)
            if len(text) > 100 and self._is_devanagari_text(text):
                paragraphs.append(text)

        return paragraphs

    def _is_devanagari_text(self, text: str) -> bool:
        """Check if text contains Devanagari characters"""
        devanagari_count = sum(1 for char in text
                              if self.bhojpuri_unicode_start <= ord(char) <= self.bhojpuri_unicode_end)
        # At least 30% of text should be Devanagari characters
        return devanagari_count / len(text) > 0.3 if len(text) > 0 else False

    def scrape_hindi_wikipedia(self) -> Generator:
        """Scrape Hindi Wikipedia with expanded article list"""
        logger.info("Starting Hindi Wikipedia scraping (Devanagari)...")

        base_url = "https://hi.wikipedia.org"

        # Expanded list of high-content Wikipedia articles
        articles_to_scrape = [
            "/wiki/मुख्य_पृष्ठ",
            "/wiki/भारत",
            "/wiki/हिन्दी_भाषा",
            "/wiki/उत्तर_प्रदेश",
            "/wiki/बिहार",
            "/wiki/साहित्य",
            "/wiki/संस्कृति",
            "/wiki/धर्म",
            "/wiki/इतिहास",
            "/wiki/विज्ञान",
            "/wiki/तकनीकी",
            "/wiki/खेल",
            "/wiki/संगीत",
            "/wiki/कला",
            "/wiki/नृत्य",
            "/wiki/राजनीति",
            "/wiki/अर्थशास्त्र",
            "/wiki/भोजन",
            "/wiki/परिवार",
            "/wiki/शिक्षा",
            "/wiki/स्वास्थ्य",
            "/wiki/पर्यावरण",
            "/wiki/कानून",
            "/wiki/समाज",
            "/wiki/परिवहन",
            "/wiki/मीडिया",
            "/wiki/तकनीकि",
            "/wiki/कम्प्यूटर",
            "/wiki/इंटरनेट",
            "/wiki/सॉफ्टवेयर",
            "/wiki/वास्तुकला",
            "/wiki/चिकित्सा",
            "/wiki/पशु",
            "/wiki/पौधे",
            "/wiki/भूगोल",
            "/wiki/खेती",
            "/wiki/व्यापार",
            "/wiki/यातायात",
            "/wiki/नैतिकता",
            "/wiki/वित्त",
            "/wiki/बैंकिंग",
            "/wiki/बीमा",
        ]

        for article in articles_to_scrape:
            if article in self.seen_urls:
                continue

            url = urljoin(base_url, article)
            self.seen_urls.add(article)

            logger.info(f"Scraping: {url}")
            try:
                paragraphs = self.extract_bhojpuri_text_wikipedia(url)

                for para in paragraphs:
                    text_hash = hashlib.md5(para.encode()).hexdigest()
                    if text_hash not in self.seen_texts:
                        self.seen_texts.add(text_hash)
                        yield para
            except Exception as e:
                logger.warning(f"Error scraping {article}: {e}")

            time.sleep(1)

    def scrape_news_sites(self) -> Generator:
        """Scrape Hindi news and content sites with Devanagari text"""
        logger.info("Starting news sites scraping (Devanagari)...")

        sources = [
            {
                'name': 'bbc_hindi',
                'url': 'https://www.bbc.com/hindi',
            },
            {
                'name': 'aajtak',
                'url': 'https://www.aajtak.in',
            },
            {
                'name': 'hindustan_times',
                'url': 'https://www.hindustantimes.com/india',
            },
            {
                'name': 'theindian_express',
                'url': 'https://indianexpress.com',
            },
            {
                'name': 'ndtv',
                'url': 'https://www.ndtv.com',
            },
        ]

        for source in sources:
            logger.info(f"Scraping {source['name']}...")

            try:
                content = self.fetch_url(source['url'], timeout=15)
                if not content:
                    logger.warning(f"No content from {source['name']}")
                    continue

                soup = BeautifulSoup(content, 'html.parser')

                # Extract all paragraphs and articles
                for elem in soup.find_all(['p', 'article', 'div', 'span']):
                    text = elem.get_text(strip=True)
                    if len(text) > 100 and self._is_devanagari_text(text):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(3)
            except Exception as e:
                logger.warning(f"Error scraping {source['name']}: {e}")

    def scrape_common_crawl_equivalents(self) -> Generator:
        """Scrape from blogs, literature, and content sites"""
        logger.info("Starting content site scraping...")

        content_sources = [
            "https://www.hindi.webdunia.com/",
            "https://www.bhaskar.com/",
            "https://www.khaskhabar.com/",
            "https://hindi.ndtv.com/",
            "https://www.amar.ujala.com/",
        ]

        for url in content_sources:
            logger.info(f"Scraping content from: {url}")
            try:
                content = self.fetch_url(url, timeout=15)
                if not content:
                    logger.warning(f"No content from {url}")
                    continue

                soup = BeautifulSoup(content, 'html.parser')

                # Extract all text elements
                for elem in soup.find_all(['p', 'div', 'article', 'span', 'li']):
                    text = elem.get_text(strip=True)
                    if len(text) > 80 and self._is_devanagari_text(text):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(3)
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")

    def save_raw_text(self, text: str, batch_id: int) -> None:
        """Save raw text to file"""
        file_path = self.raw_dir / f"batch_{batch_id:04d}.txt"
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

    def save_json_batch(self, texts: List[str], batch_id: int) -> None:
        """Save batch of texts as JSONL"""
        file_path = self.processed_dir / f"batch_{batch_id:04d}.jsonl"

        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                json.dump({'text': text}, f, ensure_ascii=False)
                f.write('\n')

    def run_scraper(self, target_paragraphs: int = 50000) -> None:
        """Run the complete scraping pipeline"""
        logger.info(f"Starting Bhojpuri scraper with target of {target_paragraphs} paragraphs")

        paragraph_count = 0
        batch_size = 100
        current_batch = []
        batch_id = 0

        start_time = time.time()

        # Combine multiple sources
        all_sources = [
            self.scrape_hindi_wikipedia(),
            self.scrape_news_sites(),
            self.scrape_common_crawl_equivalents(),
        ]

        for source in all_sources:
            try:
                for paragraph in source:
                    current_batch.append(paragraph)
                    self.save_raw_text(paragraph, batch_id)

                    if len(current_batch) >= batch_size:
                        self.save_json_batch(current_batch, batch_id)
                        logger.info(f"Batch {batch_id} saved. Total paragraphs: {paragraph_count + len(current_batch)}")
                        paragraph_count += len(current_batch)
                        current_batch = []
                        batch_id += 1

                    if paragraph_count >= target_paragraphs:
                        break
            except Exception as e:
                logger.error(f"Error in scraping: {e}")

        # Save remaining batch
        if current_batch:
            self.save_json_batch(current_batch, batch_id)
            paragraph_count += len(current_batch)

        elapsed_time = time.time() - start_time

        # Calculate statistics
        total_size = sum(f.stat().st_size for f in self.processed_dir.glob('*.jsonl'))

        logger.info(f"\n{'='*50}")
        logger.info(f"Scraping completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total paragraphs collected: {paragraph_count}")
        logger.info(f"Total data size: {total_size / 1024 / 1024:.2f} MB")
        logger.info(f"Data saved in: {self.output_dir}")
        logger.info(f"{'='*50}\n")

if __name__ == "__main__":
    scraper = BhojpuriTextScraper()
    scraper.run_scraper(target_paragraphs=50000)
