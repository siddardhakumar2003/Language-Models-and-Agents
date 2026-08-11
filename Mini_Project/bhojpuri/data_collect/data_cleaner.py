import re
import json
import logging
from pathlib import Path
from typing import List, Generator, Dict
import unicodedata
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BhojpuriDataCleaner:
    def __init__(self, input_dir: str = "./data/raw", output_dir: str = "./data/cleaned"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Devanagari Unicode range (used for Bhojpuri/Hindi)
        self.devanagari_start = 0x0900
        self.devanagari_end = 0x097F

        # Approved characters for LM training
        self.approved_punct = set('.,"!?- ')

        # Tracking statistics
        self.stats = {
            'total_processed': 0,
            'kept': 0,
            'rejected_pre_strip_ratio': 0,
            'rejected_citation_markers': 0,
            'rejected_min_length': 0,
            'rejected_min_words': 0,
            'rejected_space_density': 0,
            'rejected_exact_dup': 0,
            'rejected_near_dup': 0,
            'rejected_invalid_chars': 0,
        }

    def _is_devanagari_char(self, char: str) -> bool:
        """Check if character is Devanagari Unicode"""
        return self.devanagari_start <= ord(char) <= self.devanagari_end

    def _compute_devanagari_ratio(self, text: str) -> float:
        """Compute ratio of Devanagari characters in text"""
        if not text:
            return 0.0
        devanagari_count = sum(1 for c in text if self._is_devanagari_char(c))
        return devanagari_count / len(text)

    def _normalize_text(self, text: str) -> str:
        """Unicode normalization (NFC)"""
        return unicodedata.normalize("NFC", text)

    def _strip_citation_markers(self, text: str) -> str:
        """Remove Wikipedia citation markers like [1] [5] etc."""
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'[\[\]]', '', text)
        return text

    def _strip_zero_width_chars(self, text: str) -> str:
        """Remove zero-width and control characters"""
        text = re.sub(r'[​‌‍‎‏﻿]', '', text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text

    def _whitelist_characters(self, text: str) -> str:
        """Keep only Devanagari + digits + approved punctuation + whitespace"""
        result = []
        for char in text:
            if self._is_devanagari_char(char) or char.isdigit() or char in self.approved_punct:
                result.append(char)
        return ''.join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize multiple spaces to single space, strip leading/trailing"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _compute_space_density(self, text: str) -> float:
        """Return ratio of spaces to total characters"""
        if not text:
            return 0.0
        space_count = text.count(' ')
        return space_count / len(text)

    def clean_text(self, text: str) -> str:
        """
        Pipeline: normalize → strip citations → strip zero-width →
        whitelist → normalize spaces
        """
        # 1. Normalize Unicode
        text = self._normalize_text(text)

        # 2. Strip citation markers [5] etc
        text = self._strip_citation_markers(text)

        # 3. Strip zero-width/control chars
        text = self._strip_zero_width_chars(text)

        # 4. Whitelist characters (Devanagari + digits + approved punctuation)
        text = self._whitelist_characters(text)

        # 5. Normalize whitespace
        text = self._normalize_whitespace(text)

        return text

    def is_valid_for_training(self, text: str) -> tuple[bool, str]:
        """
        Multi-filter validation. Returns (is_valid, rejection_reason).
        """
        if not text or len(text.strip()) == 0:
            return False, 'empty'

        # Pre-strip Devanagari ratio check
        pre_strip_ratio = self._compute_devanagari_ratio(text)
        if pre_strip_ratio < 0.60:
            return False, 'pre_strip_ratio'

        # Min length (characters)
        if len(text) < 40:
            return False, 'min_length'

        # Min word count
        words = text.split()
        if len(words) < 5:
            return False, 'min_words'

        # Space density check
        space_density = self._compute_space_density(text)
        if space_density < 0.05:
            return False, 'space_density'

        return True, None

    def process_jsonl_file(self, file_path: Path) -> Generator[tuple[str, str, int], None, None]:
        """Process JSONL file and yield (cleaned_text, reason, char_len) tuples"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')

                        if not text:
                            yield None, 'empty_json_field', 0
                            continue

                        original = text
                        char_len_original = len(original)

                        # Clean the text
                        cleaned = self.clean_text(original)

                        # Validate cleaned text
                        is_valid, rejection_reason = self.is_valid_for_training(cleaned)

                        if is_valid:
                            yield cleaned, None, len(cleaned)
                        else:
                            yield None, rejection_reason, 0

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error in {file_path}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    def deduplicate_texts(self, texts: List[str]) -> tuple[List[str], int]:
        """
        Deduplication: exact match (MD5) + near-match (normalized form).
        """
        exact_seen = set()
        normalized_seen = set()
        unique = []
        dup_count = 0

        for text in texts:
            # Exact match check
            exact_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if exact_hash in exact_seen:
                dup_count += 1
                continue
            exact_seen.add(exact_hash)

            # Near-match check
            normalized = re.sub(r'[\s\.,!?\-"\']+', '', text).lower()
            near_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            if near_hash in normalized_seen:
                dup_count += 1
                continue
            normalized_seen.add(near_hash)

            unique.append(text)

        return unique, dup_count

    def run_cleaning(self, batch_size: int = 1000) -> None:
        """Run complete cleaning pipeline"""
        logger.info("Starting data cleaning...")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")

        jsonl_files = sorted(self.input_dir.glob("*.jsonl"))

        if not jsonl_files:
            logger.error(f"No JSONL files found in {self.input_dir}")
            return

        logger.info(f"Found {len(jsonl_files)} files to process")

        current_batch = []
        batch_id = 0
        total_kept = 0
        total_rejected = 0
        total_exact_dups = 0
        total_near_dups = 0

        for file_path in jsonl_files:
            logger.info(f"Processing {file_path.name}...")

            for cleaned_text, rejection_reason, char_len in self.process_jsonl_file(file_path):
                self.stats['total_processed'] += 1

                if rejection_reason:
                    total_rejected += 1
                    key = f'rejected_{rejection_reason}'
                    if key in self.stats:
                        self.stats[key] += 1
                else:
                    current_batch.append(cleaned_text)

                    if len(current_batch) >= batch_size:
                        unique_batch, dup_removed = self.deduplicate_texts(current_batch)
                        total_exact_dups += dup_removed

                        if unique_batch:
                            self._save_cleaned_batch(unique_batch, batch_id)
                            total_kept += len(unique_batch)
                            batch_id += 1

                        current_batch = []

        # Save remaining batch
        if current_batch:
            unique_batch, dup_removed = self.deduplicate_texts(current_batch)
            total_exact_dups += dup_removed

            if unique_batch:
                self._save_cleaned_batch(unique_batch, batch_id)
                total_kept += len(unique_batch)

        # Save cleaning report
        self._save_cleaning_report(total_kept, total_rejected, total_exact_dups, total_near_dups)

        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"Cleaning completed")
        logger.info(f"Total records processed: {self.stats['total_processed']}")
        logger.info(f"Total kept: {total_kept}")
        logger.info(f"Total rejected: {total_rejected}")
        logger.info(f"  - Pre-strip ratio: {self.stats['rejected_pre_strip_ratio']}")
        logger.info(f"  - Min length: {self.stats['rejected_min_length']}")
        logger.info(f"  - Min words: {self.stats['rejected_min_words']}")
        logger.info(f"  - Space density: {self.stats['rejected_space_density']}")
        logger.info(f"Duplicates removed (exact): {total_exact_dups}")
        logger.info(f"Final files: {len(list(self.output_dir.glob('cleaned_*.jsonl')))}")
        logger.info(f"Saved in: {self.output_dir}")
        logger.info(f"{'='*70}\n")

    def _save_cleaned_batch(self, texts: List[str], batch_id: int) -> None:
        """Save cleaned batch as JSONL"""
        output_file = self.output_dir / f"cleaned_{batch_id:05d}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                tokens = max(1, len(text) // 4)  # Rough token estimate
                json.dump({
                    'text': text,
                    'tokens': tokens,
                    'char_len': len(text)
                }, f, ensure_ascii=False)
                f.write('\n')

    def _save_cleaning_report(self, total_kept: int, total_rejected: int,
                            total_exact_dups: int, total_near_dups: int) -> None:
        """Save detailed cleaning report"""
        report = {
            'total_processed': self.stats['total_processed'],
            'total_kept': total_kept,
            'total_rejected': total_rejected,
            'rejection_breakdown': {
                'pre_strip_ratio': self.stats['rejected_pre_strip_ratio'],
                'min_length': self.stats['rejected_min_length'],
                'min_words': self.stats['rejected_min_words'],
                'space_density': self.stats['rejected_space_density'],
            },
            'duplicates': {
                'exact_match': total_exact_dups,
                'near_match': total_near_dups,
            }
        }

        report_file = self.output_dir / 'cleaning_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    cleaner = BhojpuriDataCleaner()
    cleaner.run_cleaning()
