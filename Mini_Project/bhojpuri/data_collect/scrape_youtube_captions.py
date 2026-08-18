#!/usr/bin/env python3
"""
Scrape YouTube captions from legitimate Bhojpuri sources.

Automatically searches YouTube for Bhojpuri content, extracts available captions,
cleans, deduplicates, and merges into corpus.

Uses youtube-search-python for discovery + youtube-transcript-api for captions.

Usage:
    python3 -m bhojpuri.data_collect.scrape_youtube_captions [--data-dir DATA_DIR] [--target-videos N]
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

# Try imports
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False

try:
    from yt_dlp import YoutubeDL
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

# Legitimate Bhojpuri YouTube search queries (native speakers, cultural content)
BHOJPURI_SEARCH_QUERIES = [
    "भोजपुरी गाना",           # Bhojpuri songs
    "भोजपुरी संगीत",          # Bhojpuri music
    "भोजपुरी फिल्म",          # Bhojpuri films
    "भोजपुरी नाटक",           # Bhojpuri drama
    "भोजपुरी समाचार",         # Bhojpuri news
    "भोजपुरी लोक गीत",        # Bhojpuri folk songs
    "भोजपुरी बोली",           # Bhojpuri language/dialect
    "भोजपुरी संस्कृति",        # Bhojpuri culture
]


class YouTubeCaptionScraper:
    """Scrape YouTube captions from legitimate Bhojpuri sources."""

    def __init__(self, data_dir: Path = None, state_file: Path = None):
        if not TRANSCRIPT_API_AVAILABLE:
            raise ImportError("youtube-transcript-api not installed")

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "youtube_raw"
        self.raw_dir.mkdir(exist_ok=True)

        if state_file is None:
            state_file = self.data_dir / "youtube_captions_state.json"
        self.state_file = Path(state_file)

        self.state = self._load_state()
        self.batch_num = self.state.get("batch_num", 0)
        self.devanagari_ratio_threshold = 0.30

    def _load_state(self) -> Dict:
        """Load scraping state."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "video_ids_processed": [],
            "batch_num": 0,
            "captions_extracted": 0,
            "queries_processed": []
        }

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

    def search_youtube(self, query: str, max_results: int = 50) -> List[str]:
        """Search YouTube for videos matching query."""
        logger.info(f"Searching YouTube for: '{query}'")

        try:
            # Use yt-dlp if available for better search
            if YTDLP_AVAILABLE:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': 'in_playlist',
                }
                with YoutubeDL(ydl_opts) as ydl:
                    search_url = f"ytsearch{max_results}:{query}"
                    info = ydl.extract_info(search_url, download=False)
                    video_ids = [entry['id'] for entry in info.get('entries', [])]
                    logger.info(f"  Found {len(video_ids)} videos")
                    return video_ids
        except Exception as e:
            logger.debug(f"yt-dlp search failed: {e}")

        # Fallback: manual search with requests
        logger.warning("Using fallback search (limited results)")
        return []

    def extract_caption(self, video_id: str) -> str:
        """Extract caption from YouTube video."""
        try:
            # List available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None

            # Priority: Bhojpuri > Hindi > Auto-generated
            for t in transcript_list.manually_created_transcripts:
                if any(x in t.language.lower() for x in ['bho', 'bhojpuri']):
                    transcript = t
                    break

            if not transcript:
                for t in transcript_list.manually_created_transcripts:
                    if any(x in t.language.lower() for x in ['hi', 'hindi']):
                        transcript = t
                        break

            if not transcript:
                for t in transcript_list.generated_transcripts:
                    if any(x in t.language.lower() for x in ['hi', 'hindi', 'bho']):
                        transcript = t
                        break

            if not transcript:
                # Get first available
                if transcript_list.manually_created_transcripts:
                    transcript = transcript_list.manually_created_transcripts[0]
                elif transcript_list.generated_transcripts:
                    transcript = transcript_list.generated_transcripts[0]

            if not transcript:
                return None

            # Fetch caption data
            caption_data = transcript.fetch()
            full_text = " ".join([entry['text'] for entry in caption_data])

            return full_text if len(full_text) > 100 else None

        except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
            logger.debug(f"Could not extract caption for {video_id}: {e}")
            return None

    def run_scraping(self, target_videos: int = 200) -> Dict[str, int]:
        """Run YouTube caption scraping."""
        logger.info("=" * 70)
        logger.info("YOUTUBE BHOJPURI CAPTION SCRAPING")
        logger.info("=" * 70)

        if not TRANSCRIPT_API_AVAILABLE:
            logger.error("youtube-transcript-api not installed")
            return {"videos_processed": 0, "captions_extracted": 0}

        if not YTDLP_AVAILABLE:
            logger.warning("yt-dlp not installed. Installing for YouTube search...")
            import subprocess
            subprocess.run(["python3", "-m", "pip", "install", "yt-dlp", "-q", "--break-system-packages"],
                          capture_output=True)
            try:
                from yt_dlp import YoutubeDL
                logger.info("✓ yt-dlp installed")
            except ImportError:
                logger.error("Failed to install yt-dlp. Skipping search.")
                return {"videos_processed": 0, "captions_extracted": 0}

        total_extracted = 0
        batch_texts = []
        batch_size = 50
        videos_per_query = max(10, target_videos // len(BHOJPURI_SEARCH_QUERIES))

        for query in BHOJPURI_SEARCH_QUERIES:
            if total_extracted >= target_videos:
                break

            if query in self.state.get("queries_processed", []):
                logger.info(f"Skipping already-processed query: {query}")
                continue

            # Search for videos
            video_ids = self.search_youtube(query, max_results=videos_per_query)

            if not video_ids:
                logger.warning(f"  No videos found for '{query}'")
                self.state["queries_processed"].append(query)
                self._save_state()
                continue

            # Extract captions from each video
            for video_id in video_ids:
                if total_extracted >= target_videos:
                    break

                if video_id in self.state.get("video_ids_processed", []):
                    logger.debug(f"Skipping already-processed video: {video_id}")
                    continue

                logger.debug(f"Extracting captions from {video_id}...")
                caption = self.extract_caption(video_id)

                if caption and self._is_devanagari_text(caption):
                    batch_texts.append({
                        "text": caption,
                        "source": "YouTube",
                        "query": query,
                        "video_id": video_id
                    })
                    total_extracted += 1
                    self.state["video_ids_processed"].append(video_id)

                    if len(batch_texts) >= batch_size:
                        batch_file = self.raw_dir / f"youtube_batch_{self.batch_num:05d}.jsonl"
                        with open(batch_file, 'w', encoding='utf-8') as f:
                            for t in batch_texts:
                                f.write(json.dumps({"text": t["text"]}, ensure_ascii=False) + "\n")
                        logger.info(f"  Wrote batch {self.batch_num}: {len(batch_texts)} captions")
                        self.batch_num += 1
                        batch_texts = []

                time.sleep(0.5)  # Rate limiting

            self.state["queries_processed"].append(query)
            self._save_state()

        # Write final batch
        if batch_texts:
            batch_file = self.raw_dir / f"youtube_batch_{self.batch_num:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for t in batch_texts:
                    f.write(json.dumps({"text": t["text"]}, ensure_ascii=False) + "\n")
            logger.info(f"  Wrote final batch {self.batch_num}: {len(batch_texts)} captions")

        self.state["captions_extracted"] = total_extracted
        self._save_state()

        logger.info(f"Total captions extracted: {total_extracted}")
        return {"videos_processed": sum(len(self.search_youtube(q, 1)) for q in BHOJPURI_SEARCH_QUERIES),
                "captions_extracted": total_extracted}


def process_youtube_captions(data_dir: Path, target_videos: int = 200) -> Dict[str, int]:
    """Complete YouTube caption scraping pipeline."""
    logger.info("=" * 70)
    logger.info("YOUTUBE CAPTIONS PIPELINE (Legitimate Bhojpuri Sources)")
    logger.info("=" * 70)

    if not TRANSCRIPT_API_AVAILABLE:
        logger.error("youtube-transcript-api required: pip install youtube-transcript-api")
        return {"videos_processed": 0, "captions_extracted": 0}

    # Scrape captions
    scraper = YouTubeCaptionScraper(data_dir=data_dir)
    scrape_stats = scraper.run_scraping(target_videos=target_videos)

    raw_dir = data_dir / "youtube_raw"
    cleaned_dir = data_dir / "youtube_cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    if not list(raw_dir.glob("*.jsonl")):
        logger.warning("No YouTube captions extracted")
        return {"videos_processed": 0, "captions_extracted": 0}

    # Clean
    logger.info("Cleaning YouTube captions...")
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
    logger.info(f"YouTube caption processing complete: {stats}")
    logger.info("=" * 70)

    return {**stats, **scrape_stats}


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube captions from Bhojpuri sources")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory"
    )
    parser.add_argument(
        "--target-videos",
        type=int,
        default=200,
        help="Target number of videos (default: 200)"
    )
    args = parser.parse_args()

    process_youtube_captions(args.data_dir, target_videos=args.target_videos)

    # Update config
    config_path = args.data_dir / "config.json"
    update_config_bhojpuri(config_path, args.data_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    yt_source = "YouTube Captions (Bhojpuri songs, films, cultural content - auto-extracted)"
    if yt_source not in config.get("data_sources", []):
        config["data_sources"].append(yt_source)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"Token progress: {config['token_progress']['progress_str']}")


if __name__ == "__main__":
    main()
