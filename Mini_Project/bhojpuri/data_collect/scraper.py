import os
import json
import time
import requests
from datetime import datetime
from typing import List, Generator, Set
from pathlib import Path
import logging
from urllib.parse import urljoin, urlparse, quote
from bs4 import BeautifulSoup
import hashlib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BhojpuriTextScraper:
    def __init__(self, output_dir: str = None):
        # If no output_dir specified, use bhojpuri/data/ (script location aware)
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent.parent / "data")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.state_file = self.output_dir / "scrape_state.json"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Mount HTTP adapter with retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.seen_urls: Set[str] = set()
        self.seen_texts: Set[str] = set()
        self.token_count = 0
        self.paragraph_count = 0
        self.next_batch_id = 0

        # Bhojpuri Unicode range (Devanagari script)
        self.bhojpuri_unicode_start = 0x0900
        self.bhojpuri_unicode_end = 0x097F

        self.load_state()

    def load_state(self) -> None:
        """Load checkpointed state to resume scraping"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.seen_urls = set(state.get('seen_urls', []))
                self.seen_texts = set(state.get('seen_texts', []))
                self.paragraph_count = state.get('paragraph_count', 0)
                self.next_batch_id = state.get('next_batch_id', 0)
                logger.info(f"Loaded state: {len(self.seen_urls)} URLs, {len(self.seen_texts)} texts, {self.paragraph_count} paragraphs, next batch {self.next_batch_id}")
            except Exception as e:
                logger.warning(f"Error loading state: {e}, starting fresh")

    def save_state(self) -> None:
        """Save current scraping state for resumability"""
        state = {
            'seen_urls': list(self.seen_urls),
            'seen_texts': list(self.seen_texts),
            'paragraph_count': self.paragraph_count,
            'next_batch_id': self.next_batch_id,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def fetch_url(self, url: str, timeout: int = 15) -> str:
        """Fetch content from URL with automatic retries"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching {url}: {e}")
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

    def discover_wikipedia_titles(self, target_count: int = 800, api_base: str = "https://hi.wikipedia.org/w/api.php") -> List[str]:
        """Discover Hindi Wikipedia article titles using MediaWiki API"""
        logger.info(f"Discovering Wikipedia titles (target: {target_count})...")

        discovered = set()
        seed_categories = [
            'वर्ग:हिन्दी_साहित्य',
            'वर्ग:भारत',
            'वर्ग:उत्तर_प्रदेश',
            'वर्ग:बिहार',
        ]

        # Random sampling for broad coverage
        for attempt in range(3):
            try:
                params = {
                    'action': 'query',
                    'list': 'random',
                    'rnnamespace': '0',
                    'rnlimit': '500',
                    'format': 'json'
                }
                response = self.session.get(api_base, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('query', {}).get('random', []):
                        title = item.get('title')
                        if title:
                            discovered.add(title)
                    if len(discovered) >= target_count:
                        break
            except Exception as e:
                logger.warning(f"Error in random sampling (attempt {attempt+1}): {e}")
            time.sleep(1)

        # Seed categories for topical coherence
        for category in seed_categories:
            if len(discovered) >= target_count:
                break
            try:
                cmcontinue = None
                for _ in range(2):
                    params = {
                        'action': 'query',
                        'list': 'categorymembers',
                        'cmtitle': category,
                        'cmlimit': '500',
                        'cmnamespace': '0',
                        'format': 'json'
                    }
                    if cmcontinue:
                        params['cmcontinue'] = cmcontinue

                    response = self.session.get(api_base, params=params, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('query', {}).get('categorymembers', []):
                            title = item.get('title')
                            if title:
                                discovered.add(title)
                        cmcontinue = data.get('query-continue', {}).get('categorymembers', {}).get('cmcontinue')
                        if not cmcontinue or len(discovered) >= target_count:
                            break
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"Error discovering category {category}: {e}")

        result = list(discovered)[:target_count]
        logger.info(f"Discovered {len(result)} Wikipedia article titles")
        return result

    def scrape_hindi_wikipedia(self) -> Generator:
        """Scrape discovered Hindi Wikipedia articles (Devanagari proxy for Bhojpuri)"""
        logger.info("Starting Hindi Wikipedia scraping (Devanagari proxy)...")

        titles = self.discover_wikipedia_titles(target_count=800)

        for title in titles:
            url = f"https://hi.wikipedia.org/wiki/{quote(title)}"

            if url in self.seen_urls:
                continue

            self.seen_urls.add(url)

            try:
                paragraphs = self.extract_bhojpuri_text_wikipedia(url)

                for para in paragraphs:
                    text_hash = hashlib.md5(para.encode()).hexdigest()
                    if text_hash not in self.seen_texts:
                        self.seen_texts.add(text_hash)
                        yield para
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")

            time.sleep(0.5)

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
        """Run the complete scraping pipeline with checkpointing"""
        logger.info(f"Starting Bhojpuri scraper with target of {target_paragraphs} paragraphs (resuming from {self.paragraph_count})")

        batch_size = 100
        current_batch = []

        start_time = time.time()

        # Combine multiple sources
        all_sources = [
            ("Hindi Wikipedia", self.scrape_hindi_wikipedia()),
            ("News Sites", self.scrape_news_sites()),
            ("Content Sites", self.scrape_common_crawl_equivalents()),
        ]

        for source_name, source in all_sources:
            logger.info(f"Processing source: {source_name}")
            try:
                for paragraph in source:
                    current_batch.append(paragraph)
                    self.save_raw_text(paragraph, self.next_batch_id)

                    if len(current_batch) >= batch_size:
                        self.save_json_batch(current_batch, self.next_batch_id)
                        self.paragraph_count += len(current_batch)
                        self.next_batch_id += 1

                        # Save state after each batch
                        self.save_state()

                        logger.info(f"Batch {self.next_batch_id-1} saved. Total: {self.paragraph_count} paragraphs [{source_name}]")
                        current_batch = []

                        if self.paragraph_count >= target_paragraphs:
                            logger.info("Target reached!")
                            break
            except Exception as e:
                logger.error(f"Error in source {source_name}: {e}")

            if self.paragraph_count >= target_paragraphs:
                break

        # Save remaining batch
        if current_batch:
            self.save_json_batch(current_batch, self.next_batch_id)
            self.paragraph_count += len(current_batch)
            self.next_batch_id += 1
            self.save_state()

        elapsed_time = time.time() - start_time

        # Calculate statistics
        total_size = sum(f.stat().st_size for f in self.processed_dir.glob('*.jsonl')) if self.processed_dir.exists() else 0
        total_files = len(list(self.processed_dir.glob('*.jsonl'))) if self.processed_dir.exists() else 0

        logger.info(f"\n{'='*70}")
        logger.info(f"Scraping completed in {elapsed_time / 60:.2f} minutes")
        logger.info(f"Total paragraphs collected: {self.paragraph_count}")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total data size: {total_size / 1024 / 1024:.2f} MB")
        if elapsed_time > 0:
            logger.info(f"Average paragraphs per second: {self.paragraph_count / elapsed_time:.0f}")
        logger.info(f"Data saved in: {self.output_dir}")
        logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    scraper = BhojpuriTextScraper()
    scraper.run_scraper(target_paragraphs=50000)
