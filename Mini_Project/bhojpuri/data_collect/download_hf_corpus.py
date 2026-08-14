#!/usr/bin/env python3
"""
Download & integrate HuggingFace BhojpuriCorpus into the Bhojpuri training dataset.

Pipeline:
  1. Download parquet from HF → save as jsonl (386K docs, ~25M tokens)
  2. Clean + validate with BhojpuriDataCleaner (identical rules as phase1b)
  3. Global dedup: remove exact/near duplicates vs. existing bhoj.txt + internal
  4. Split 80/10/10 (seed 42, by whole line) → append to train/val/test/bhoj.txt
  5. Update config.json with new token_progress/splits
  6. Archive directories to prevent re-processing
"""

import json
import logging
import re
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set
import requests
import pandas as pd

from .data_cleaner import BhojpuriDataCleaner
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_PARQUET_URL = "https://huggingface.co/datasets/Satyam810/BhojpuriCorpus/resolve/main/data/BhojpuriCorpus.parquet"
HF_DATASET_NAME = "Satyam810/BhojpuriCorpus"

class HFBhojpuriDownloader:
    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.hf_raw_dir = self.data_dir / "hf_raw"
        self.hf_cleaned_dir = self.data_dir / "hf_cleaned"
        self.hf_raw_download_dir = self.data_dir / "hf_raw_download"

    def download_parquet(self) -> Path:
        """Download parquet from HuggingFace."""
        self.hf_raw_download_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = self.hf_raw_download_dir / "BhojpuriCorpus.parquet"

        if parquet_path.exists():
            logger.info(f"Parquet already exists: {parquet_path}, skipping download")
            return parquet_path

        logger.info(f"Downloading {HF_DATASET_NAME} parquet (~295MB)...")
        try:
            resp = requests.get(HF_PARQUET_URL, stream=True, timeout=60)
            resp.raise_for_status()
            total_size = int(resp.headers.get('content-length', 0))
            downloaded = 0

            with open(parquet_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (10*1024*1024) == 0:
                            pct = 100.0 * downloaded / total_size
                            logger.info(f"  Downloaded {downloaded/1e6:.1f}MB / {total_size/1e6:.1f}MB ({pct:.1f}%)")

            logger.info(f"✓ Downloaded {parquet_path.stat().st_size / 1e6:.1f}MB")
            return parquet_path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def parquet_to_jsonl(self, parquet_path: Path) -> int:
        """Convert parquet to jsonl in hf_raw directory."""
        self.hf_raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Reading parquet: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} documents")

        # Validate schema
        if 'text' not in df.columns:
            raise ValueError(f"Missing 'text' column. Columns: {df.columns.tolist()}")

        # Write to jsonl in batches
        batch_size = 20000
        total_written = 0
        batch_id = 0

        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            batch_file = self.hf_raw_dir / f"hf_batch_{batch_id:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for _, row in batch_df.iterrows():
                    text = row['text']
                    if text:
                        json.dump({'text': text}, f, ensure_ascii=False)
                        f.write('\n')
                        total_written += 1

            logger.info(f"Wrote batch {batch_id}: {batch_end - batch_start} records → {batch_file.name}")
            batch_id += 1

        logger.info(f"✓ Converted {total_written} documents to jsonl in {self.hf_raw_dir}")
        return total_written

    def clean_validate_manually(self) -> Tuple[List[str], Dict]:
        """
        Clean + validate texts manually (don't use run_cleaning, so we can track and control dedup).
        Returns list of cleaned texts and a stats dict.
        """
        cleaner = BhojpuriDataCleaner(input_dir=str(self.hf_raw_dir), output_dir=str(self.hf_cleaned_dir))
        stats = {
            'total_processed': 0,
            'kept': 0,
            'rejected_reasons': {},
            'batches_saved': 0,
        }

        cleaned_texts = []
        batch_id = 0

        for jsonl_file in sorted(self.hf_raw_dir.glob("hf_batch_*.jsonl")):
            logger.info(f"Cleaning {jsonl_file.name}...")

            for cleaned_text, rejection_reason, char_len in cleaner.process_jsonl_file(jsonl_file):
                stats['total_processed'] += 1

                if rejection_reason:
                    reason_key = f"rejected_{rejection_reason}"
                    stats['rejected_reasons'][reason_key] = stats['rejected_reasons'].get(reason_key, 0) + 1
                else:
                    cleaned_texts.append(cleaned_text)

        stats['kept'] = len(cleaned_texts)
        logger.info(f"Cleaned: {stats['kept']} kept, {stats['total_processed'] - stats['kept']} rejected")

        # Save cleaned texts to cleaned_*.jsonl files
        batch_size = 1000
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i+batch_size]
            batch_file = self.hf_cleaned_dir / f"cleaned_{batch_id:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for text in batch:
                    tokens = max(1, len(text) // 4)
                    json.dump({'text': text, 'tokens': tokens, 'char_len': len(text)}, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"Saved {batch_file.name}: {len(batch)} texts")
            batch_id += 1

        stats['batches_saved'] = batch_id
        return cleaned_texts, stats

    def deduplicate_with_existing_corpus(self, new_texts: List[str]) -> Tuple[List[str], Dict]:
        """
        Dedup: remove exact/near duplicates of existing bhoj.txt + internal duplicates.
        Returns unique texts and dedup stats.
        """
        logger.info("Seeding dedup with existing bhoj.txt...")
        existing_file = self.data_dir / "bhoj.txt"
        exact_hashes = set()
        normalized_hashes = set()

        if existing_file.exists():
            line_count = 0
            with open(existing_file, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.rstrip('\n').strip()
                    if not text:
                        continue

                    # Seed exact-match hashes
                    exact_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    exact_hashes.add(exact_hash)

                    # Seed near-match hashes
                    normalized = re.sub(r'[\s\.,!?\-"\']+', '', text).lower()
                    near_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
                    normalized_hashes.add(near_hash)

                    line_count += 1

            logger.info(f"Seeded dedup with {line_count} existing lines")
        else:
            logger.warning(f"No existing bhoj.txt found at {existing_file}")

        # Now filter new texts through seeded sets
        unique_texts = []
        exact_dup_count = 0
        near_dup_count = 0

        logger.info(f"Deduplicating {len(new_texts)} new texts...")
        for text in new_texts:
            # Exact match
            exact_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if exact_hash in exact_hashes:
                exact_dup_count += 1
                continue
            exact_hashes.add(exact_hash)

            # Near match
            normalized = re.sub(r'[\s\.,!?\-"\']+', '', text).lower()
            near_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            if near_hash in normalized_hashes:
                near_dup_count += 1
                continue
            normalized_hashes.add(near_hash)

            unique_texts.append(text)

        logger.info(f"Dedup result: {len(unique_texts)} unique (removed {exact_dup_count} exact + {near_dup_count} near dups)")

        return unique_texts, {
            'exact_duplicates': exact_dup_count,
            'near_duplicates': near_dup_count,
            'unique_texts': len(unique_texts),
        }

    def rewrite_cleaned_jsonl_deduped(self, unique_texts: List[str]) -> None:
        """
        Overwrite cleaned_*.jsonl files with deduplicated texts only.
        """
        logger.info(f"Rewriting cleaned_*.jsonl with {len(unique_texts)} deduplicated texts...")

        # Clear old cleaned files
        for f in self.hf_cleaned_dir.glob("cleaned_*.jsonl"):
            f.unlink()

        batch_size = 1000
        batch_id = 0
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i+batch_size]
            batch_file = self.hf_cleaned_dir / f"cleaned_{batch_id:05d}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for text in batch:
                    tokens = max(1, len(text) // 4)
                    json.dump({'text': text, 'tokens': tokens, 'char_len': len(text)}, f, ensure_ascii=False)
                    f.write('\n')
            batch_id += 1

    def run(self) -> Dict:
        """Execute the full pipeline."""
        logger.info("=" * 70)
        logger.info("HuggingFace BhojpuriCorpus Integration")
        logger.info("=" * 70)

        # Step A: Download
        parquet_path = self.download_parquet()

        # Step B: Convert to jsonl
        total_raw = self.parquet_to_jsonl(parquet_path)

        # Step C: Clean + validate
        cleaned_texts, clean_stats = self.clean_validate_manually()
        logger.info(f"After cleaning: {clean_stats['kept']} kept")
        for reason, count in clean_stats['rejected_reasons'].items():
            logger.info(f"  {reason}: {count}")

        # Step D: Global dedup (including existing corpus)
        unique_texts, dedup_stats = self.deduplicate_with_existing_corpus(cleaned_texts)
        logger.info(f"After dedup: {dedup_stats['unique_texts']} unique")
        logger.info(f"  Exact duplicates (existing): {dedup_stats['exact_duplicates']}")
        logger.info(f"  Near duplicates (existing): {dedup_stats['near_duplicates']}")

        # Step E: Rewrite cleaned jsonl with deduplicated texts
        self.rewrite_cleaned_jsonl_deduped(unique_texts)

        # Step F: Split + append using the existing merge_ocr_into_splits
        logger.info("Splitting and appending to train/val/test/bhoj.txt...")
        merge_result = merge_ocr_into_splits(
            self.data_dir,
            accumulator_filename="bhoj.txt",
            split_filename="bhoj.txt",
            cleaned_dir_name="hf_cleaned",
            temp_batch_filename="hf_new_batch.txt",
        )

        # Step G: Update config
        logger.info("Updating config.json...")
        update_config_bhojpuri(str(self.data_dir / "config.json"), str(self.data_dir))

        # Append HF to data_sources
        config_path = self.data_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            hf_source = "Satyam810/BhojpuriCorpus (HuggingFace: 386K docs, ~25M tokens, deduplicated)"
            if "data_sources" not in config:
                config["data_sources"] = []
            if hf_source not in config["data_sources"]:
                config["data_sources"].append(hf_source)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✓ Added HF source to config.json")

        # Step H: Archive hf_raw
        archive_raw = self.data_dir / f"hf_raw_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.hf_raw_dir.exists():
            logger.info(f"Archiving {self.hf_raw_dir.name} → {archive_raw.name}")
            shutil.move(str(self.hf_raw_dir), str(archive_raw))

        # Clean up parquet download
        if self.hf_raw_download_dir.exists():
            logger.info(f"Cleaning up {self.hf_raw_download_dir.name}")
            shutil.rmtree(self.hf_raw_download_dir)

        logger.info("=" * 70)
        logger.info("✓ HuggingFace BhojpuriCorpus integration complete!")
        logger.info("=" * 70)

        return {
            'raw_documents': total_raw,
            'cleaned_kept': clean_stats['kept'],
            'cleaned_rejected': clean_stats['total_processed'] - clean_stats['kept'],
            'dedup_unique': dedup_stats['unique_texts'],
            'dedup_exact_removed': dedup_stats['exact_duplicates'],
            'dedup_near_removed': dedup_stats['near_duplicates'],
            'merge_result': merge_result,
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download & integrate HuggingFace BhojpuriCorpus")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory (default: bhojpuri/data/)")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parent.parent / "data"

    downloader = HFBhojpuriDownloader(args.data_dir)
    result = downloader.run()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Raw documents:          {result['raw_documents']}")
    print(f"After cleaning:         {result['cleaned_kept']} kept, {result['cleaned_rejected']} rejected")
    print(f"After dedup:            {result['dedup_unique']} unique")
    print(f"  Duplicates removed:   {result['dedup_exact_removed']} (exact) + {result['dedup_near_removed']} (near)")
    print(f"Merge result:")
    print(f"  Total lines split:    {result['merge_result'].get('total_lines', 0)}")
    print(f"  Train:                +{result['merge_result'].get('train_lines', 0)}")
    print(f"  Val:                  +{result['merge_result'].get('val_lines', 0)}")
    print(f"  Test:                 +{result['merge_result'].get('test_lines', 0)}")
    print(f"  Accumulator:          {result['merge_result'].get('accumulator_before', 0)} → {result['merge_result'].get('accumulator_after', 0)} lines")
    print("=" * 70)

if __name__ == "__main__":
    main()
