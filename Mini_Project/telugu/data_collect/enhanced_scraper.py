import os
import json
import time
import requests
from datetime import datetime
from typing import List, Generator, Set, Dict
from pathlib import Path
import logging
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
import hashlib
import re
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedTeluguScraper:
    def __init__(self, output_dir: str = None):
        # If no output_dir specified, use telugu/data/ (script location aware)
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent.parent / "data")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
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

        self.seen_texts: Set[str] = set()
        self.token_count = 0
        self.target_tokens = 500_000_000
        self.scraped_urls: Set[str] = set()
        self.next_batch_id = 0

        self.load_state()

    def load_state(self) -> None:
        """Load checkpointed state to resume scraping"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.scraped_urls = set(state.get('scraped_urls', []))
                self.seen_texts = set(state.get('seen_texts', []))
                self.token_count = state.get('token_count', 0)
                self.next_batch_id = state.get('next_batch_id', 0)
                logger.info(f"Loaded state: {len(self.scraped_urls)} URLs, {len(self.seen_texts)} texts, {self.token_count} tokens, next batch {self.next_batch_id}")
            except Exception as e:
                logger.warning(f"Error loading state: {e}, starting fresh")

    def save_state(self) -> None:
        """Save current scraping state for resumability"""
        state = {
            'scraped_urls': list(self.scraped_urls),
            'seen_texts': list(self.seen_texts),
            'token_count': self.token_count,
            'next_batch_id': self.next_batch_id,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)"""
        return max(1, len(text) // 4)

    def fetch_url(self, url: str, timeout: int = 15) -> str:
        """Fetch content from URL with automatic retries and exponential backoff"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching {url}: {e}")
            return ""

    def _is_telugu_text(self, text: str, min_length: int = 20) -> bool:
        """Check if text contains Telugu characters"""
        if not text or len(text) < min_length:
            return False

        telugu_unicode_start = 0x0C00
        telugu_unicode_end = 0x0C7F

        telugu_count = sum(1 for char in text if telugu_unicode_start <= ord(char) <= telugu_unicode_end)
        return telugu_count / len(text) > 0.25

    def discover_wikipedia_titles(self, target_count: int = 3000, api_base: str = "https://te.wikipedia.org/w/api.php") -> List[str]:
        """Discover Telugu Wikipedia article titles using MediaWiki API"""
        logger.info(f"Discovering Wikipedia titles (target: {target_count})...")

        discovered = set()
        seed_categories = [
            'వర్గం:తెలుగు_సాహిత్యం',
            'వర్గం:ఆంధ్రప్రదేశ్',
            'వర్గం:తెలంగాణ',
            'వర్గం:భారతదేశ_చరిత్ర',
        ]

        # Random sampling for broad coverage
        for attempt in range(5):
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
                for _ in range(3):  # Max 3 continuation tokens per category
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

    def scrape_wikipedia_telugu(self) -> Generator:
        """Scrape discovered Telugu Wikipedia articles"""
        logger.info("Starting Wikipedia Telugu scraping with discovered titles...")

        titles = self.discover_wikipedia_titles(target_count=3000)

        for title in titles:
            url = f"https://te.wikipedia.org/wiki/{quote(title)}"

            if url in self.scraped_urls:
                continue

            self.scraped_urls.add(url)

            try:
                content = self.fetch_url(url)
                if not content:
                    continue

                soup = BeautifulSoup(content, 'html.parser')
                main_content = soup.find('div', id='mw-content-text')
                if not main_content:
                    main_content = soup

                # Remove unwanted sections
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

                for p in main_content.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and self._is_telugu_text(text):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")

    def discover_wiki_subpages(self, project: str = "wikibooks", api_base: str = None) -> List[str]:
        """Discover subpages in a wiki project using list=allpages"""
        if api_base is None:
            api_base = f"https://te.{project}.org/w/api.php"

        logger.info(f"Discovering {project} pages...")
        pages = []

        try:
            apfrom = None
            for attempt in range(3):  # Max 3 continuation tokens
                params = {
                    'action': 'query',
                    'list': 'allpages',
                    'aplimit': '500',
                    'apnamespace': '0',
                    'format': 'json'
                }
                if apfrom:
                    params['apfrom'] = apfrom

                response = self.session.get(api_base, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('query', {}).get('allpages', []):
                        title = item.get('title')
                        if title:
                            pages.append(title)
                    apfrom = data.get('query-continue', {}).get('allpages', {}).get('apfrom')
                    if not apfrom:
                        break
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Error discovering {project} pages: {e}")

        logger.info(f"Discovered {len(pages)} {project} pages")
        return pages

    def scrape_wikibooks_telugu(self) -> Generator:
        """Scrape Telugu Wikibooks using discovered pages"""
        logger.info("Starting Wikibooks Telugu scraping...")

        pages = self.discover_wiki_subpages(project="wikibooks", api_base="https://te.wikibooks.org/w/api.php")

        for title in pages[:500]:  # Limit to avoid excessive scraping
            url = f"https://te.wikibooks.org/wiki/{quote(title)}"

            if url in self.scraped_urls:
                continue

            self.scraped_urls.add(url)

            try:
                content = self.fetch_url(url)
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

                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error scraping wikibook {url}: {e}")

    def scrape_wikiquote_telugu(self) -> Generator:
        """Scrape Telugu Wikiquote"""
        logger.info("Starting Wikiquote Telugu scraping...")

        pages = self.discover_wiki_subpages(project="wikiquote", api_base="https://te.wikiquote.org/w/api.php")

        for title in pages[:300]:
            url = f"https://te.wikiquote.org/wiki/{quote(title)}"

            if url in self.scraped_urls:
                continue

            self.scraped_urls.add(url)

            try:
                content = self.fetch_url(url)
                if not content:
                    continue

                soup = BeautifulSoup(content, 'html.parser')

                for quote in soup.find_all(['p', 'blockquote', 'div']):
                    text = quote.get_text(strip=True)
                    if text and self._is_telugu_text(text, min_length=10):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash not in self.seen_texts:
                            self.seen_texts.add(text_hash)
                            yield text

                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error scraping wikiquote {url}: {e}")

    def scrape_wikisource_telugu(self) -> Generator:
        """Scrape Telugu Wikisource"""
        logger.info("Starting Wikisource Telugu scraping...")

        pages = self.discover_wiki_subpages(project="wikisource", api_base="https://te.wikisource.org/w/api.php")

        for title in pages[:300]:
            url = f"https://te.wikisource.org/wiki/{quote(title)}"

            if url in self.scraped_urls:
                continue

            self.scraped_urls.add(url)

            try:
                content = self.fetch_url(url)
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

                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error scraping wikisource {url}: {e}")

    def scrape_news_sites(self) -> Generator:
        """Scrape Telugu news websites"""
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

                for elem in soup.find_all(['nav', 'footer', 'aside']):
                    elem.decompose()

                for elem in soup.find_all(['div', 'section']):
                    class_str = (elem.get('class', []) or []) if elem.get('class') else []
                    id_str = (elem.get('id') or '').lower()
                    class_lower = ' '.join(class_str).lower() if class_str else ''

                    if any(pat in class_lower or pat in id_str for pat in skip_patterns):
                        elem.decompose()

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
        logger.info(f"Starting scraper. Target: {max_tokens / 1e6:.1f}M tokens (resuming from {self.token_count / 1e6:.2f}M)")

        current_batch = []
        self.target_tokens = max_tokens

        start_time = time.time()

        sources = [
            ("Wikipedia", self.scrape_wikipedia_telugu()),
            ("Wikibooks", self.scrape_wikibooks_telugu()),
            ("Wikiquote", self.scrape_wikiquote_telugu()),
            ("Wikisource", self.scrape_wikisource_telugu()),
            ("News Sites", self.scrape_news_sites()),
        ]

        for source_name, source_generator in sources:
            logger.info(f"Processing source: {source_name}")

            try:
                for text in source_generator:
                    if not text or len(text) < 20:
                        continue

                    current_batch.append(text)

                    if len(current_batch) >= batch_size:
                        batch_tokens = self.save_batch(current_batch, self.next_batch_id)
                        self.token_count += batch_tokens
                        self.next_batch_id += 1

                        # Save state after each batch
                        self.save_state()

                        progress_pct = (self.token_count / max_tokens) * 100
                        logger.info(
                            f"Batch {self.next_batch_id-1}: {batch_tokens/1e6:.3f}M tokens | "
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
            batch_tokens = self.save_batch(current_batch, self.next_batch_id)
            self.token_count += batch_tokens
            self.next_batch_id += 1
            self.save_state()

        elapsed_time = time.time() - start_time

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
