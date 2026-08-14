#!/usr/bin/env python3
import json, logging, os, re, shutil, time
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Dict, Generator, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BhojpuriArchiveOrgDownloader:
    SCRAPE_API_URL = "https://archive.org/services/search/v1/scrape"
    METADATA_API_URL = "https://archive.org/metadata"
    DOWNLOAD_URL = "https://archive.org/download"

    SEARCH_STRATEGIES = [
        "language:(Hindi) AND mediatype:(texts) AND format:(DjVuTXT)",
        "subject:(Hindi OR Bhojpuri) AND mediatype:(texts)",
        "creator:(Hindi) OR language:(Hindi)",
        "mediatype:(texts) AND (title:(hindi OR bhojpuri) OR description:(hindi OR bhojpuri))",
    ]

    def __init__(self, language_query="Hindi", script_range=(0x0900, 0x097F), output_dir=None,
                 state_file=None, batch_prefix="arch", batch_size=50, max_concurrency=6, request_timeout=30):
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent.parent / "data" / "ocr_raw")
        if state_file is None:
            state_file = str(Path(__file__).resolve().parent.parent / "data" / "archive_org_state.json")

        self.language_query = language_query
        self.script_range = script_range
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
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, starting fresh")
        return {"downloaded_identifiers": {}, "search_cursor": None, "next_batch_id": 0, "items_exhausted": False,
                "current_strategy_idx": 0, "exhausted_strategies": []}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def search_page(self, cursor=None, count=100):
        try:
            strategy_idx = self.state.get("current_strategy_idx", 0)
            if strategy_idx >= len(self.SEARCH_STRATEGIES):
                logger.info("All search strategies exhausted")
                return {"items": [], "cursor": None, "total": 0}

            query = self.SEARCH_STRATEGIES[strategy_idx]
            params = {"q": query, "fields": "identifier,title", "count": count}
            if cursor:
                params["cursor"] = cursor
            logger.info(f"Searching archive.org [strategy {strategy_idx+1}/{len(self.SEARCH_STRATEGIES)}] (cursor={cursor[:20] if cursor else 'START'}...)...")
            resp = self.session.get(self.SCRAPE_API_URL, params=params, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            return {"items": data.get("items", []), "cursor": data.get("cursor"), "total": data.get("total", 0)}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"items": [], "cursor": None, "total": 0}

    def iter_new_identifiers(self):
        while True:
            strategy_idx = self.state.get("current_strategy_idx", 0)
            if strategy_idx >= len(self.SEARCH_STRATEGIES):
                logger.info("All search strategies exhausted")
                self.state["items_exhausted"] = True
                self._save_state()
                break

            cursor = self.state.get("search_cursor")
            page = self.search_page(cursor=cursor, count=100)
            for item in page.get("items", []):
                identifier = item.get("identifier", "").strip()
                if identifier and identifier not in self.state["downloaded_identifiers"]:
                    yield identifier
            cursor = page.get("cursor")
            if not cursor:
                logger.info(f"Strategy {strategy_idx+1} exhausted, moving to next")
                self.state["search_cursor"] = None
                self.state["current_strategy_idx"] = strategy_idx + 1
                self._save_state()
                continue
            self.state["search_cursor"] = cursor
            self._save_state()

    def fetch_item_metadata(self, identifier):
        try:
            resp = self.session.get(f"{self.METADATA_API_URL}/{identifier}", timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            for f in data.get("files", []):
                if f.get("name", "").endswith("_djvu.txt"):
                    return {"identifier": identifier, "title": data.get("metadata", {}).get("title", ""),
                            "djvu_filename": f.get("name")}
            return None
        except Exception as e:
            logger.debug(f"{identifier}: metadata fetch failed: {e}")
            return None

    def download_djvu_text(self, identifier, djvu_filename):
        try:
            resp = self.session.get(f"{self.DOWNLOAD_URL}/{identifier}/{djvu_filename}", timeout=self.request_timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"{identifier}: download failed: {e}")
            return None

    def _preclean_djvu_text(self, raw_text):
        text = re.sub(r'\n{2,}', '\n\n', raw_text)
        lines = re.split(r'\n{2,}', text)
        cleaned = []
        for page in lines:
            page = re.sub(r'(\w+)-\n(\w+)', r'\1\2', page)
            page = re.sub(r'\n+', ' ', page)
            page = re.sub(r' +', ' ', page)
            if page.strip():
                cleaned.append(page.strip())
        return cleaned

    def _estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def _is_script_text(self, text):
        if not text or len(text) < 20:
            return False
        script_count = sum(1 for c in text if self.script_range[0] <= ord(c) <= self.script_range[1])
        return (script_count / len(text)) >= 0.30

    def _save_batch(self, records, batch_id):
        batch_file = self.output_dir / f"{self.batch_prefix}_batch_{batch_id:06d}.jsonl"
        with open(batch_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Saved batch {batch_id}: {len(records)} records")

    def run(self, target_new_items=None, max_runtime_seconds=None):
        start_time = time.time()
        items_downloaded, items_failed, raw_tokens = 0, 0, 0
        batch_id = self.state.get("next_batch_id", 0)
        batch = []
        logger.info("=" * 70)
        logger.info(f"Archive.org Downloader: {self.language_query} (Bhojpuri proxy)")
        logger.info("=" * 70)
        for identifier in self.iter_new_identifiers():
            if target_new_items and items_downloaded >= target_new_items:
                break
            if max_runtime_seconds and (time.time() - start_time) > max_runtime_seconds:
                break
            meta = self.fetch_item_metadata(identifier)
            if not meta:
                items_failed += 1
                continue
            text = self.download_djvu_text(identifier, meta["djvu_filename"])
            if not text:
                items_failed += 1
                continue
            precleaned_lines = self._preclean_djvu_text(text)
            for line in precleaned_lines:
                if self._is_script_text(line):
                    line_md5 = md5(line.encode()).hexdigest()
                    if line_md5 not in self.state["downloaded_identifiers"]:
                        tokens = self._estimate_tokens(line)
                        batch.append({"text": line, "tokens": tokens, "timestamp": datetime.now().isoformat(),
                                     "source_identifier": identifier})
                        raw_tokens += tokens
            self.state["downloaded_identifiers"][identifier] = {"batch_id": batch_id, "tokens": sum(
                r.get("tokens", 0) for r in batch if r.get("source_identifier") == identifier),
                                                                 "timestamp": datetime.now().isoformat()}
            items_downloaded += 1
            if len(batch) >= self.batch_size:
                self._save_batch(batch, batch_id)
                batch_id += 1
                batch = []
            time.sleep(0.5)
        if batch:
            self._save_batch(batch, batch_id)
            batch_id += 1
        self.state["next_batch_id"] = batch_id
        self._save_state()
        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"Downloaded: {items_downloaded}, Failed: {items_failed}, Raw tokens: {raw_tokens}")
        logger.info(f"Elapsed: {elapsed:.1f}s, Items exhausted: {self.state['items_exhausted']}")
        logger.info("=" * 70)
        return {"items_downloaded": items_downloaded, "items_failed": items_failed,
                "raw_tokens_estimate": raw_tokens, "items_exhausted": self.state.get("items_exhausted", False),
                "elapsed_seconds": elapsed}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download books from archive.org (Bhojpuri)")
    parser.add_argument("--target-new-items", type=int, default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    downloader = BhojpuriArchiveOrgDownloader(state_file=args.state_file, output_dir=args.output_dir)
    result = downloader.run(target_new_items=args.target_new_items)
    return 0 if result["items_downloaded"] > 0 else 1

if __name__ == "__main__":
    exit(main())
