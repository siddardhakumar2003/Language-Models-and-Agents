#!/usr/bin/env python3
"""
Phase 3 Translation Orchestrator: Hindi → Bhojpuri via machine translation.
Resumable, incremental background job (like Phase 1b).
"""
import json
import logging
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .download_archive_org_hindi_books import download_hindi_books
from .hindi_to_bhojpuri_translator_nllb import HindiToBhojpuriTranslator
from .news_crawler import BhojpuriNewsCrawler
from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase3TranslationOrchestrator:
    """Orchestrates Hindi→Bhojpuri translation pipeline in resumable cycles."""

    def __init__(
        self,
        data_dir=None,
        hindi_articles_per_cycle=50,
        hindi_books_per_cycle=10,
        translations_per_cycle=100,
        cycle_delay_seconds=30,
        target_tokens=500_000_000
    ):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")

        self.data_dir = Path(data_dir)
        self.hindi_articles_per_cycle = hindi_articles_per_cycle
        self.hindi_books_per_cycle = hindi_books_per_cycle
        self.translations_per_cycle = translations_per_cycle
        self.cycle_delay_seconds = cycle_delay_seconds
        self.target_tokens = target_tokens
        self.stop_requested = False
        self.log_file = self.data_dir / "phase3_translation_progress.log"
        self.report_file = self.data_dir / "phase3_translation_report.json"
        self.status_file = self.data_dir / "phase3_live_status.json"
        self.pre_tokens_at_start = None
        self.cumulative_translations = 0
        self.cumulative_api_calls = 0

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _handle_sigterm(self, sig, frame):
        self._log_progress("INTERRUPTED by signal", silent=True)
        self.stop_requested = True

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

    def _archive_cleaned_dir(self, cleaned_dir_name: str):
        """Archive cleaned directory after merge."""
        cleaned_dir = self.data_dir / cleaned_dir_name
        if cleaned_dir.exists() and list(cleaned_dir.glob("*")):
            archive_dir = self.data_dir / f"{cleaned_dir_name}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(cleaned_dir), str(archive_dir))
            logger.info(f"Archived {cleaned_dir_name} to {archive_dir.name}")

    def run_cycle(self, cycle_num: int):
        """Execute one complete cycle of the translation pipeline."""
        try:
            corpus_tokens_before = self._get_corpus_tokens()
            corpus_lines_before = sum(1 for _ in open(self.data_dir / "bhoj.txt")) if (self.data_dir / "bhoj.txt").exists() else 0
            self._log_progress(f"CYCLE {cycle_num} | Starting... (corpus: {corpus_tokens_before/1e6:.1f}M, {corpus_lines_before:,} lines)")
            cycle_start_time = time.time()

            # Step 1: Gather Hindi news
            logger.info("Step 1: Gathering Hindi news articles...")
            news_crawler = BhojpuriNewsCrawler(
                output_dir=str(self.data_dir / "hindi_news_raw"),
                state_file=str(self.data_dir / "news_crawl_translate_state.json"),
                batch_prefix="hindi_news"
            )
            news_result = news_crawler.run(target_new_articles=self.hindi_articles_per_cycle)
            news_added = news_result.get('articles_fetched', 0)

            # Step 2: Gather Hindi books from archive.org
            logger.info("Step 2: Downloading Hindi books from archive.org...")
            books_result = download_hindi_books(
                data_dir=str(self.data_dir),
                target_items=self.hindi_books_per_cycle
            )
            books_added = books_result.get('items_downloaded', 0)

            # Step 3: Translate gathered Hindi text to Bhojpuri
            logger.info("Step 3: Translating Hindi → Bhojpuri...")
            translator = HindiToBhojpuriTranslator(data_dir=str(self.data_dir))
            source_dirs = [
                str(self.data_dir / "hindi_news_raw"),
                str(self.data_dir / "hindi_books_raw"),
                str(self.data_dir / "hindi_wiki_raw")
            ]
            trans_result = translator.run(
                source_raw_dirs=source_dirs,
                target_new_translations=self.translations_per_cycle
            )
            translations_done = trans_result.get('translations_done', 0)
            trans_stats = trans_result.get('stats', {})
            translations_failed = trans_stats.get('total_records_failed', 0)

            # Step 4: Clean translated text
            logger.info("Step 4: Cleaning translated text...")
            cleaner = BhojpuriDataCleaner(
                input_dir=str(self.data_dir / "hindi_translated_raw"),
                output_dir=str(self.data_dir / "hindi_translated_cleaned")
            )
            cleaner.run_cleaning()
            report_file = self.data_dir / "hindi_translated_cleaned" / "cleaning_report.json"
            texts_cleaned = 0
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    texts_cleaned = report.get('total_kept', 0)

            # Step 5: Deduplicate against existing corpus
            logger.info("Step 5: Deduplicating against existing corpus...")
            # Use temp dir to avoid file I/O bug with same source/output
            with tempfile.TemporaryDirectory() as tmp:
                temp_dedup = Path(tmp) / "hindi_translated_dedup"
                temp_dedup.mkdir()
                deduplicate_with_existing_corpus(
                    data_dir=self.data_dir,
                    source_cleaned_dir=self.data_dir / "hindi_translated_cleaned",
                    output_cleaned_dir=temp_dedup
                )
                # Move deduplicated files back to original location
                shutil.rmtree(self.data_dir / "hindi_translated_cleaned")
                shutil.move(str(temp_dedup), str(self.data_dir / "hindi_translated_cleaned"))

            # Step 6: Merge into train/val/test splits
            logger.info("Step 6: Merging into splits...")
            merge_result = merge_ocr_into_splits(
                self.data_dir,
                accumulator_filename="bhoj.txt",
                split_filename="bhoj.txt",
                cleaned_dir_name="hindi_translated_cleaned",
                temp_batch_filename="hindi_translated_new_batch.txt"
            )
            merged_lines = merge_result.get('total_lines', 0)
            self._archive_cleaned_dir("hindi_translated_cleaned")

            # Step 7: Update config
            logger.info("Step 7: Updating config...")
            update_config_bhojpuri(str(self.data_dir / "config.json"), self.data_dir)

            # Patch in hindi_translation section
            config_path = self.data_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                if "hindi_translation" not in config:
                    config["hindi_translation"] = {}

                config["hindi_translation"]["source"] = "Hindi news (news_crawler.py) + archive.org Hindi books + hi.wikipedia.org (hiwiki dump)"
                config["hindi_translation"]["translation_method"] = "NLLB-200 distilled-600M (local CPU inference, hin_Deva→bho_Deva)"
                config["hindi_translation"]["hindi_records_translated"] = config["hindi_translation"].get("hindi_records_translated", 0) + translations_done
                config["hindi_translation"]["hindi_records_failed"] = config["hindi_translation"].get("hindi_records_failed", 0) + translations_failed
                config["hindi_translation"]["translated_lines_kept_after_cleaning"] = config["hindi_translation"].get("translated_lines_kept_after_cleaning", 0) + texts_cleaned
                config["hindi_translation"]["api_calls_total"] = config["hindi_translation"].get("api_calls_total", 0) + trans_stats.get('total_api_calls', 0)
                config["hindi_translation"]["last_run_at"] = datetime.now().isoformat()

                # Add to data_sources if not already present (dedup by exact match)
                # Remove old Google Translate entry if present
                old_google_str = "Hindi→Bhojpuri Machine Translation (Google Translate unofficial endpoint, hi→bho; sources: Hindi news + archive.org Hindi public-domain books)"
                new_source_str = "Hindi→Bhojpuri Machine Translation (NLLB-200, local CPU; sources: Hindi news + archive.org books + hi.wikipedia.org)"
                if "data_sources" in config:
                    if old_google_str in config["data_sources"]:
                        config["data_sources"].remove(old_google_str)
                    if new_source_str not in config["data_sources"]:
                        config["data_sources"].append(new_source_str)
                else:
                    config["data_sources"] = [new_source_str]

                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

            # Log cycle completion with corpus growth
            corpus_tokens_after = self._get_corpus_tokens()
            corpus_lines_after = sum(1 for _ in open(self.data_dir / "bhoj.txt")) if (self.data_dir / "bhoj.txt").exists() else 0
            corpus_tokens_gained = corpus_tokens_after - corpus_tokens_before
            corpus_lines_gained = corpus_lines_after - corpus_lines_before
            progress_pct = 100.0 * corpus_tokens_after / self.target_tokens
            cycle_time = time.time() - cycle_start_time

            self._log_progress(
                f"CYCLE {cycle_num:3d} | news:+{news_added:3d} books:+{books_added:3d} translated:+{translations_done:3d} "
                f"(failed:{translations_failed:2d}) cleaned:+{texts_cleaned:3d} merged:+{merged_lines:4d} | "
                f"CORPUS: {corpus_tokens_before/1e6:7.1f}M → {corpus_tokens_after/1e6:7.1f}M "
                f"(+{corpus_tokens_gained/1e6:6.2f}M, +{corpus_lines_gained:5d} lines) | "
                f"Progress: {progress_pct:5.1f}% ({corpus_tokens_after/1e6:7.1f}M/{self.target_tokens/1e6:7.1f}M) | "
                f"time:{cycle_time:6.1f}s"
            )

            # Write live-status JSON for consistent monitoring
            self.cumulative_translations += translations_done
            self.cumulative_api_calls += trans_stats.get('total_api_calls', 0)
            avg_tokens_per_cycle = (corpus_tokens_after - self.pre_tokens_at_start) / cycle_num if self.pre_tokens_at_start else 0

            status = {
                "updated_at": datetime.now().isoformat(timespec='seconds'),
                "cycle_num": cycle_num,
                "corpus_tokens": corpus_tokens_after,
                "corpus_lines": corpus_lines_after,
                "target_tokens": self.target_tokens,
                "progress_percent": round(progress_pct, 2),
                "tokens_gained_this_cycle": corpus_tokens_gained,
                "lines_gained_this_cycle": corpus_lines_gained,
                "cumulative_translations": self.cumulative_translations,
                "cumulative_api_calls": self.cumulative_api_calls,
                "avg_tokens_gained_per_cycle": int(avg_tokens_per_cycle),
                "cycles_since_start": cycle_num,
                "news_added": news_added,
                "books_added": books_added,
                "translated_this_cycle": translations_done,
                "failed_this_cycle": translations_failed,
                "cleaned_this_cycle": texts_cleaned,
                "merged_this_cycle": merged_lines,
                "cycle_time_seconds": round(cycle_time, 1)
            }

            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)

            return {
                "success": True,
                "target_reached": corpus_tokens_after >= self.target_tokens,
                "corpus_tokens": corpus_tokens_after,
                "news_added": news_added,
                "books_added": books_added,
                "translations_done": translations_done,
                "texts_cleaned": texts_cleaned,
                "merged_lines": merged_lines
            }

        except Exception as e:
            self._log_progress(f"CYCLE {cycle_num} ERROR | {str(e)}", silent=True)
            logger.exception(f"Cycle {cycle_num} failed")
            raise

    def _write_report(self, pre_tokens, post_tokens, cycles_completed, total_translations):
        """Write final report in same schema as phase2_expansion_report.json."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "pre_expansion": {
                "corpus_tokens": pre_tokens,
                "formatted": f"{pre_tokens/1e6:.1f}M"
            },
            "post_expansion": {
                "corpus_tokens": post_tokens,
                "formatted": f"{post_tokens/1e6:.1f}M"
            },
            "expansion_delta": {
                "corpus_tokens": post_tokens - pre_tokens,
                "formatted": f"{(post_tokens - pre_tokens)/1e6:.1f}M",
                "percentage": 100.0 * (post_tokens - pre_tokens) / pre_tokens if pre_tokens > 0 else 0
            },
            "cycles_completed": cycles_completed,
            "total_translations": total_translations,
            "target_tokens": self.target_tokens,
            "final_progress_percentage": 100.0 * post_tokens / self.target_tokens
        }

        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report written to {self.report_file}")

    def run(self):
        """Main loop: run cycles until target reached or stop requested."""
        self._log_progress("Phase 3 Translation Orchestrator started", silent=True)
        pre_tokens = self._get_corpus_tokens()
        self.pre_tokens_at_start = pre_tokens
        cycle_num = 0
        no_progress_count = 0
        max_no_progress_cycles = 5
        last_corpus_tokens = pre_tokens
        total_translations = 0

        try:
            while not self.stop_requested:
                cycle_num += 1
                try:
                    result = self.run_cycle(cycle_num)
                    total_translations += result.get("translations_done", 0)

                    if result["target_reached"]:
                        self._log_progress(
                            f"STOPPED reason=target_reached corpus={result['corpus_tokens']/1e6:.1f}M/"
                            f"{self.target_tokens/1e6:.1f}M cycles={cycle_num}",
                            silent=True
                        )
                        break

                    if result["corpus_tokens"] == last_corpus_tokens:
                        no_progress_count += 1
                        if no_progress_count >= max_no_progress_cycles:
                            self._log_progress(
                                f"STOPPED reason=no_progress cycles_without_progress={no_progress_count} "
                                f"corpus={result['corpus_tokens']/1e6:.1f}M/{self.target_tokens/1e6:.1f}M",
                                silent=True
                            )
                            break
                    else:
                        no_progress_count = 0
                        last_corpus_tokens = result["corpus_tokens"]

                    if not self.stop_requested:
                        time.sleep(self.cycle_delay_seconds)

                except KeyboardInterrupt:
                    self._log_progress("Interrupted by user", silent=True)
                    self.stop_requested = True
                    break
                except Exception as e:
                    logger.error(f"Error in cycle {cycle_num}: {e}")
                    self._log_progress(f"Cycle {cycle_num} failed: {e}", silent=True)
                    time.sleep(30)  # Wait before retry

            post_tokens = self._get_corpus_tokens()
            self._write_report(pre_tokens, post_tokens, cycle_num, total_translations)
            self._log_progress("Phase 3 Translation Orchestrator finished", silent=True)

        except Exception as e:
            logger.error(f"Fatal error in orchestrator: {e}")
            self._log_progress(f"Fatal error: {e}", silent=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 Translation Orchestrator (Hindi→Bhojpuri)")
    parser.add_argument("--hindi-articles-per-cycle", type=int, default=50)
    parser.add_argument("--hindi-books-per-cycle", type=int, default=10)
    parser.add_argument("--translations-per-cycle", type=int, default=100)
    parser.add_argument("--cycle-delay", type=int, default=30)
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    orchestrator = Phase3TranslationOrchestrator(
        data_dir=args.data_dir,
        hindi_articles_per_cycle=args.hindi_articles_per_cycle,
        hindi_books_per_cycle=args.hindi_books_per_cycle,
        translations_per_cycle=args.translations_per_cycle,
        cycle_delay_seconds=args.cycle_delay,
        target_tokens=args.target_tokens
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
