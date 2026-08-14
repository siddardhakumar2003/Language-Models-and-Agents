#!/usr/bin/env python3
import json, logging, shutil, signal, sys, time
from datetime import datetime
from pathlib import Path
from .archive_org_downloader import BhojpuriArchiveOrgDownloader
from .news_crawler import BhojpuriNewsCrawler
from .data_cleaner import BhojpuriDataCleaner
from .ocr_merge import merge_ocr_into_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase1bOrchestrator:
    def __init__(self, data_dir=None, target_tokens=500_000_000, books_per_cycle=100, articles_per_cycle=500, cycle_delay_seconds=30):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")
        self.data_dir = Path(data_dir)
        self.target_tokens = target_tokens
        self.books_per_cycle = books_per_cycle
        self.articles_per_cycle = articles_per_cycle
        self.cycle_delay_seconds = cycle_delay_seconds
        self.stop_requested = False
        self.log_file = self.data_dir / "phase1b_progress.log"
        self.cumulative_stats_file = self.data_dir / "ocr_cumulative_stats.json"
        self._load_or_init_cumulative_stats()
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _handle_sigterm(self, sig, frame):
        self._log_progress("INTERRUPTED by signal", silent=True)
        self.stop_requested = True

    def _load_or_init_cumulative_stats(self):
        if self.cumulative_stats_file.exists():
            try:
                with open(self.cumulative_stats_file, 'r') as f:
                    self.cumulative_stats = json.load(f)
            except:
                self.cumulative_stats = {"total_files_processed": 0, "total_pages_extracted": 0,
                                        "total_lines_kept": 0, "total_lines_rejected": 0}
        else:
            self.cumulative_stats = {"total_files_processed": 0, "total_pages_extracted": 0,
                                    "total_lines_kept": 0, "total_lines_rejected": 0}

    def _save_cumulative_stats(self):
        self.cumulative_stats["last_updated"] = datetime.now().isoformat()
        with open(self.cumulative_stats_file, 'w') as f:
            json.dump(self.cumulative_stats, f, indent=2)

    def _log_progress(self, message, silent=False):
        timestamp = datetime.now().isoformat(timespec='seconds')
        log_line = f"[{timestamp}] {message}"
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
        if not silent:
            logger.info(message)

    def _get_corpus_tokens(self):
        corpus_file = self.data_dir / "bhoj.txt"
        if corpus_file.exists():
            return corpus_file.stat().st_size // 4
        return 0

    def _archive_ocr_raw(self):
        ocr_raw = self.data_dir / "ocr_raw"
        if ocr_raw.exists() and list(ocr_raw.glob("*")):
            archive_dir = self.data_dir / f"ocr_raw_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(ocr_raw), str(archive_dir))
            ocr_raw.mkdir(exist_ok=True)

    def run_cycle(self, cycle_num):
        try:
            self._log_progress(f"CYCLE {cycle_num} | Starting...")
            downloader = BhojpuriArchiveOrgDownloader(output_dir=str(self.data_dir / "ocr_raw"))
            book_result = downloader.run(target_new_items=self.books_per_cycle)
            crawler = BhojpuriNewsCrawler(output_dir=str(self.data_dir / "ocr_raw"))
            news_result = crawler.run(target_new_articles=self.articles_per_cycle)
            cleaner = BhojpuriDataCleaner(input_dir=str(self.data_dir / "ocr_raw"),
                                        output_dir=str(self.data_dir / "ocr_cleaned"))
            cleaner.run_cleaning()
            report_file = self.data_dir / "ocr_cleaned" / "cleaning_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                self.cumulative_stats["total_lines_kept"] += report.get("total_kept", 0)
                self.cumulative_stats["total_lines_rejected"] += report.get("total_rejected", 0)
                texts_cleaned, texts_rejected = report.get("total_kept", 0), report.get("total_rejected", 0)
            else:
                texts_cleaned, texts_rejected = 0, 0
            merge_result = merge_ocr_into_splits(self.data_dir, accumulator_filename="bhoj.txt", split_filename="bhoj.txt")
            self._archive_ocr_raw()
            self._save_cumulative_stats()
            corpus_tokens = self._get_corpus_tokens()
            pass_rate = 100.0 * texts_cleaned / (texts_cleaned + texts_rejected) if (texts_cleaned + texts_rejected) > 0 else 0
            self._log_progress(f"CYCLE {cycle_num} | books:+{book_result['items_downloaded']} articles:+{news_result['articles_fetched']} cleaned:{texts_cleaned} rejected:{texts_rejected} ({pass_rate:.1f}%) corpus:{corpus_tokens/1e6:.1f}M/{self.target_tokens/1e6:.1f}M ({100.0*corpus_tokens/self.target_tokens:.1f}%)")
            return {"success": True, "target_reached": corpus_tokens >= self.target_tokens,
                   "sources_exhausted": book_result["items_exhausted"] and news_result["domains_exhausted"],
                   "corpus_tokens": corpus_tokens}
        except Exception as e:
            self._log_progress(f"CYCLE {cycle_num} ERROR | {str(e)}", silent=True)
            raise

    def run(self):
        self._log_progress("Phase 1b Orchestrator started", silent=True)
        cycle_num = 0
        no_progress_count = 0
        max_no_progress_cycles = 3
        last_corpus_tokens = 0
        while not self.stop_requested:
            cycle_num += 1
            result = self.run_cycle(cycle_num)
            if result["target_reached"]:
                self._log_progress(f"STOPPED reason=target_reached corpus={result['corpus_tokens']/1e6:.1f}M/{self.target_tokens/1e6:.1f}M", silent=True)
                break
            if result["corpus_tokens"] == last_corpus_tokens:
                no_progress_count += 1
                if result["sources_exhausted"] and no_progress_count >= max_no_progress_cycles:
                    self._log_progress(f"STOPPED reason=no_progress_with_exhausted_sources cycles_without_progress={no_progress_count} corpus={result['corpus_tokens']/1e6:.1f}M/{self.target_tokens/1e6:.1f}M", silent=True)
                    break
            else:
                no_progress_count = 0
                last_corpus_tokens = result["corpus_tokens"]
            time.sleep(self.cycle_delay_seconds)
        self._log_progress("Phase 1b Orchestrator finished", silent=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1b Orchestrator (Bhojpuri)")
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--books-per-cycle", type=int, default=100)
    parser.add_argument("--articles-per-cycle", type=int, default=500)
    parser.add_argument("--cycle-delay", type=int, default=30)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    orchestrator = Phase1bOrchestrator(data_dir=args.data_dir, target_tokens=args.target_tokens,
                                      books_per_cycle=args.books_per_cycle, articles_per_cycle=args.articles_per_cycle,
                                      cycle_delay_seconds=args.cycle_delay)
    orchestrator.run()

if __name__ == "__main__":
    main()
