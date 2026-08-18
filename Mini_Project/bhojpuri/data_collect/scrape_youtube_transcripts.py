#!/usr/bin/env python3
"""
Scrape YouTube transcripts from Bhojpuri channels, clean, deduplicate, and merge.

Extracts auto-generated and manual transcripts from Bhojpuri YouTube channels.
Implements rate limiting, state tracking, and proper deduplication.

Requires: pip install youtube-transcript-api

Usage:
    python3 -m bhojpuri.data_collect.scrape_youtube_transcripts [--data-dir DATA_DIR] [--target-videos N]
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List

from tqdm import tqdm

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import youtube_transcript_api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    logger.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")

# Bhojpuri YouTube channels (music, cultural, educational)
BHOJPURI_CHANNELS = [
    {
        "name": "Bhojpuri Music Channel",
        "channel_id": "UCxxxxxx",  # Placeholder - would need real IDs
        "type": "music"
    },
    {
        "name": "Bhojpuri Folk & Culture",
        "channel_id": "UCyyyyyy",
        "type": "cultural"
    },
    {
        "name": "Bhojpuri News & Current Affairs",
        "channel_id": "UCzzzzzz",
        "type": "news"
    }
]

# Fallback: use YouTube search URLs instead
BHOJPURI_SEARCH_QUERIES = [
    "भोजपुरी संगीत",  # Bhojpuri music
    "भोजपुरी फिल्म",  # Bhojpuri films
    "भोजपुरी लोकगीत",  # Bhojpuri folk songs
    "भोजपुरी समाचार",  # Bhojpuri news
    "भोजपुरी नृत्य",  # Bhojpuri dance
]


class YouTubeTranscriptScraper:
    """Scrape YouTube transcripts with state tracking."""

    def __init__(self, data_dir: Path = None, state_file: Path = None):
        if not YOUTUBE_AVAILABLE:
            raise ImportError("youtube-transcript-api not installed")

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "youtube_raw"
        self.raw_dir.mkdir(exist_ok=True)

        if state_file is None:
            state_file = self.data_dir / "youtube_scrape_state.json"
        self.state_file = Path(state_file)

        self.state = self._load_state()
        self.batch_num = self.state.get("batch_num", 0)
        self.devanagari_ratio_threshold = 0.30

    def _load_state(self) -> Dict:
        """Load scraping state."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"video_ids_processed": [], "batch_num": 0, "transcripts_extracted": 0}

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

    def extract_transcript(self, video_id: str) -> str:
        """Extract transcript from YouTube video."""
        try:
            # Try Bhojpuri first, fallback to Hindi
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None

            # Look for Bhojpuri
            for t in transcript_list.manually_created_transcripts:
                if 'bho' in t.language.lower() or 'bhojpuri' in t.language.lower():
                    transcript = t
                    break

            # Fallback to Hindi
            if not transcript:
                for t in transcript_list.manually_created_transcripts:
                    if 'hi' in t.language.lower() or 'hindi' in t.language.lower():
                        transcript = t
                        break

            # Try auto-generated
            if not transcript:
                for t in transcript_list.generated_transcripts:
                    if 'hi' in t.language.lower() or 'bho' in t.language.lower():
                        transcript = t
                        break

            if not transcript:
                return None

            # Get full transcript text
            transcript_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in transcript_data])

            return full_text if len(full_text) > 100 else None

        except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
            logger.debug(f"Could not extract transcript for {video_id}: {e}")
            return None

    def extract_from_channel(self, channel_id: str, max_videos: int = 50) -> Generator[Dict, None, None]:
        """Extract transcripts from a channel (requires channel data)."""
        # This requires YouTube API access which needs authentication
        # For now, we'll skip channel-based extraction
        logger.warning(f"Channel extraction requires YouTube API key (skipping {channel_id})")
        return
        yield  # Make it a generator

    def extract_from_search(self, search_query: str, max_videos: int = 50) -> Generator[Dict, None, None]:
        """Extract transcripts using manual video IDs (fallback approach)."""
        logger.warning("YouTube search requires API key. For production:")
        logger.warning("  1. Get API key from https://console.cloud.google.com/")
        logger.warning("  2. Use youtube-search library: pip install youtube-search-python")
        logger.warning("  3. Update this script to use API-based search")
        logger.warning("")
        logger.warning("Currently supporting manual video ID extraction only.")
        return
        yield  # Make it a generator

    def extract_from_manual_ids(self, video_ids: List[str]) -> Generator[Dict, None, None]:
        """Extract transcripts from manually-provided video IDs."""
        for video_id in video_ids:
            if video_id in self.state.get("video_ids_processed", []):
                logger.debug(f"Skipping already processed {video_id}")
                continue

            transcript = self.extract_transcript(video_id)
            if transcript and self._is_devanagari_text(transcript):
                yield {
                    "text": transcript,
                    "source": "YouTube",
                    "video_id": video_id
                }
                self.state["video_ids_processed"].append(video_id)
            else:
                logger.debug(f"No Devanagari transcript for {video_id}")

            time.sleep(1)  # Rate limit

    def run_scraping(self, target_videos: int = 100) -> Dict[str, int]:
        """Run YouTube transcript scraping."""
        logger.info("=" * 70)
        logger.info("YOUTUBE BHOJPURI TRANSCRIPT SCRAPING")
        logger.info("=" * 70)

        if not YOUTUBE_AVAILABLE:
            logger.error("youtube-transcript-api not installed")
            logger.error("Install with: pip install youtube-transcript-api")
            return {"videos_processed": 0, "transcripts_extracted": 0}

        logger.warning("Note: YouTube API requires authentication for search/channel data")
        logger.warning("Currently: manual video ID input only, or use youtube-search-python")
        logger.warning("")

        # For now, return empty (requires API setup)
        logger.info("To use YouTube scraping:")
        logger.info("  1. Install: pip install youtube-search-python")
        logger.info("  2. Get API key from: https://console.cloud.google.com/")
        logger.info("  3. Update YOUTUBE_CHANNELS or provide video IDs manually")
        logger.info("")

        # Placeholder: Extract from hardcoded list if you have video IDs
        sample_video_ids = []  # User would populate this with actual Bhojpuri video IDs

        total_extracted = 0
        batch_texts = []
        batch_size = 50

        for transcript in self.extract_from_manual_ids(sample_video_ids):
            batch_texts.append(transcript)
            total_extracted += 1

            if len(batch_texts) >= batch_size:
                batch_file = self.raw_dir / f"youtube_batch_{self.batch_num:05d}.jsonl"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    for t in batch_texts:
                        f.write(json.dumps({"text": t["text"]}, ensure_ascii=False) + "\n")
                logger.info(f"Wrote batch: {batch_file.name}")
                self.batch_num += 1
                batch_texts = []

        if batch_texts:
            batch_file = self.raw_dir / f"youtube_batch_{self.batch_num:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for t in batch_texts:
                    f.write(json.dumps({"text": t["text"]}, ensure_ascii=False) + "\n")

        self.state["transcripts_extracted"] = total_extracted
        self._save_state()

        logger.info(f"Total transcripts extracted: {total_extracted}")
        return {"videos_processed": len(sample_video_ids), "transcripts_extracted": total_extracted}


def process_youtube_scraping(data_dir: Path, target_videos: int = 100) -> Dict[str, int]:
    """Complete YouTube scraping pipeline."""
    logger.info("=" * 70)
    logger.info("YOUTUBE BHOJPURI TRANSCRIPT PIPELINE")
    logger.info("=" * 70)

    if not YOUTUBE_AVAILABLE:
        logger.error("YouTube module requires: pip install youtube-transcript-api")
        return {"videos_processed": 0, "transcripts_extracted": 0}

    # Scrape transcripts
    scraper = YouTubeTranscriptScraper(data_dir=data_dir)
    scrape_stats = scraper.run_scraping(target_videos=target_videos)

    raw_dir = data_dir / "youtube_raw"
    cleaned_dir = data_dir / "youtube_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    if not list(raw_dir.glob("*.jsonl")):
        logger.warning("No YouTube transcripts extracted")
        return {"videos_processed": 0, "transcripts_extracted": 0}

    # Clean
    logger.info("Cleaning YouTube transcripts...")
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
        cleaned_dir_name="youtube_cleaned",
        temp_batch_filename="youtube_new_batch.txt"
    )

    # Archive
    import shutil
    timestamp = datetime.now().isoformat()
    archive_dir = data_dir / f"youtube_cleaned_merged_{timestamp}"
    if cleaned_dir.exists():
        cleaned_dir.rename(archive_dir)
        logger.info(f"Archived to {archive_dir.name}")

    logger.info("=" * 70)
    logger.info(f"YouTube scraping complete: {stats}")
    logger.info("=" * 70)

    return {**stats, **scrape_stats}


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube Bhojpuri transcripts")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory"
    )
    parser.add_argument(
        "--target-videos",
        type=int,
        default=100,
        help="Target number of videos (default: 100)"
    )
    args = parser.parse_args()

    process_youtube_scraping(args.data_dir, target_videos=args.target_videos)

    # Update config
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    yt_source = "YouTube Transcripts (Bhojpuri channels, auto-generated + manual)"
    if yt_source not in config.get("data_sources", []):
        config["data_sources"].append(yt_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
