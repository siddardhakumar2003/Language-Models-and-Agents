#!/usr/bin/env python3
import json, logging, re, time, xml.etree.ElementTree as ET
from datetime import datetime
from hashlib import md5
from pathlib import Path
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeluguNewsCrawler:
    DOMAINS = [
        {"name": "eenadu", "base_url": "https://www.eenadu.net"},
        {"name": "greatandhra", "base_url": "https://www.greatandhra.com"},
        {"name": "andhrajyothy", "base_url": "https://www.andhrajyothy.com"},
        {"name": "sakshi", "base_url": "https://www.sakshi.com"},
        {"name": "ntnews", "base_url": "https://www.ntnews.com"},
        {"name": "thehindu-telugu", "base_url": "https://www.thehindu.com"},
        {"name": "deccan-chronicle", "base_url": "https://www.deccanchronicle.com"},
        {"name": "telangana-today", "base_url": "https://telanganatoday.com"},
        {"name": "tv9-telugu", "base_url": "https://www.tv9telugu.com"},
    ]

    def __init__(self, output_dir=None, state_file=None, batch_prefix="news", batch_size=50, request_timeout=15):
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent.parent / "data" / "ocr_raw")
        if state_file is None:
            state_file = str(Path(__file__).resolve().parent.parent / "data" / "news_crawl_state.json")

        self.output_dir = Path(output_dir)
        self.state_file = Path(state_file)
        self.batch_prefix = batch_prefix
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["HEAD", "GET", "OPTIONS"])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
        self.state = self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"seen_urls": {}, "seen_texts": {}, "next_batch_id": 0, "domains_completed": []}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def discover_via_sitemap(self, domain_cfg):
        urls = []
        for sitemap_path in ["/sitemap-news.xml", "/sitemap.xml"]:
            try:
                resp = self.session.get(domain_cfg["base_url"] + sitemap_path, timeout=self.request_timeout)
                root = ET.fromstring(resp.content)
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    url = url_elem.text.strip()
                    if url and url not in self.state["seen_urls"]:
                        urls.append(url)
                if urls:
                    break
            except:
                continue
        return urls

    def fetch_article(self, url):
        try:
            resp = self.session.get(url, timeout=self.request_timeout)
            html = resp.text
            for tag in ['script', 'style', 'nav', 'footer']:
                html = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', html, flags=re.DOTALL | re.IGNORECASE)
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
            text = '\n'.join(p for p in paragraphs)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'&[a-z]+;', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip() if text else None
        except:
            return None

    def _is_telugu_text(self, text):
        if not text or len(text) < 20:
            return False
        count = sum(1 for c in text if 0x0C00 <= ord(c) <= 0x0C7F)
        return (count / len(text)) >= 0.25

    def _estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def _save_batch(self, records, batch_id):
        batch_file = self.output_dir / f"{self.batch_prefix}_batch_{batch_id:06d}.jsonl"
        with open(batch_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Saved batch {batch_id}: {len(records)} records")

    def run(self, target_new_articles=None, max_runtime_seconds=None):
        start_time = time.time()
        articles_fetched, articles_failed, raw_tokens = 0, 0, 0
        batch_id = self.state.get("next_batch_id", 0)
        batch = []

        logger.info("=" * 70)
        logger.info("News Crawler: Telugu")
        logger.info("=" * 70)

        for domain_cfg in self.DOMAINS:
            if domain_cfg["name"] in self.state.get("domains_completed", []):
                continue
            if target_new_articles and articles_fetched >= target_new_articles:
                break
            if max_runtime_seconds and (time.time() - start_time) > max_runtime_seconds:
                break

            logger.info(f"\nCrawling: {domain_cfg['name']}")
            urls = self.discover_via_sitemap(domain_cfg)
            logger.info(f"  Discovered {len(urls)} URLs")

            for url in urls[:100]:
                if target_new_articles and articles_fetched >= target_new_articles:
                    break
                text = self.fetch_article(url)
                if not text or not self._is_telugu_text(text):
                    articles_failed += 1
                    continue
                text_md5 = md5(text.encode()).hexdigest()
                if text_md5 in self.state["seen_texts"]:
                    continue
                self.state["seen_texts"][text_md5] = True
                self.state["seen_urls"][url] = True
                tokens = self._estimate_tokens(text)
                batch.append({"text": text, "tokens": tokens, "timestamp": datetime.now().isoformat(), "source_url": url})
                raw_tokens += tokens
                articles_fetched += 1
                if len(batch) >= self.batch_size:
                    self._save_batch(batch, batch_id)
                    batch_id += 1
                    batch = []
                time.sleep(1.0)

            self.state["domains_completed"].append(domain_cfg["name"])
            logger.info(f"  Domain done: {articles_fetched} articles total")

        if batch:
            self._save_batch(batch, batch_id)
            batch_id += 1

        self.state["next_batch_id"] = batch_id
        self._save_state()
        elapsed = time.time() - start_time

        logger.info("=" * 70)
        logger.info(f"Fetched: {articles_fetched}, Failed: {articles_failed}, Raw tokens: {raw_tokens}")
        logger.info(f"Elapsed: {elapsed:.1f}s")
        logger.info("=" * 70)

        return {"articles_fetched": articles_fetched, "articles_failed": articles_failed,
                "raw_tokens_estimate": raw_tokens, "domains_exhausted": len(self.state["domains_completed"]) >= len(self.DOMAINS),
                "elapsed_seconds": elapsed}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crawl news sites for Telugu text")
    parser.add_argument("--target-new-articles", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    crawler = TeluguNewsCrawler(output_dir=args.output_dir)
    result = crawler.run(target_new_articles=args.target_new_articles)
    return 0 if result["articles_fetched"] > 0 else 1

if __name__ == "__main__":
    exit(main())
