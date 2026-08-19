#!/usr/bin/env python3
"""
Hindi to Bhojpuri translation via local NLLB-200 model (HuggingFace transformers).
Resumable, with state tracking. Replaces the Google Translate HTTP-based translator.
"""
import json
import logging
import os
from hashlib import md5
from pathlib import Path
from typing import Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model + tokenizer (lazy-loaded on first use)
_model = None
_tokenizer = None


def _load_model_and_tokenizer(model_name: str = "facebook/nllb-200-distilled-600M"):
    """Lazily load NLLB-200 model and tokenizer once."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    os.environ['HF_HOME'] = 'bhojpuri/model/.hf_cache'

    logger.info("Loading NLLB-200 model (HuggingFace)...")
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _model.eval()  # Set to evaluation mode for inference

    logger.info("Loading NLLB-200 tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    return _model, _tokenizer


def split_into_chunks(text, max_chars=700):
    """Split text on Devanagari sentence terminators, pack greedily up to max_chars."""
    terminators = ['।', '.', '!', '?']
    chunks = []
    current_chunk = ""

    # Split on terminators first
    parts = []
    current_part = ""
    for char in text:
        current_part += char
        if char in terminators:
            parts.append(current_part)
            current_part = ""
    if current_part:
        parts.append(current_part)

    # Pack greedily
    for part in parts:
        if len(current_chunk) + len(part) <= max_chars:
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def translate_chunk(text: str) -> Optional[str]:
    """Translate a single chunk via local NLLB-200 model."""
    try:
        model, tokenizer = _load_model_and_tokenizer()

        # Prepare input with forced_bos_token_id for target language
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

        # Generate translation
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("bho_Deva"),
            max_length=256,
            num_beams=5,
        )

        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated if translated.strip() else None

    except Exception as e:
        logger.warning(f"Translation chunk failed: {e}")
        return None


def translate_long_text(text: str) -> Optional[str]:
    """Translate long text with chunking. Batch all chunks together for efficiency."""
    try:
        model, tokenizer = _load_model_and_tokenizer()

        chunks = split_into_chunks(text, max_chars=700)
        if not chunks:
            return None

        # Tokenize all chunks and translate in one batch
        inputs = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        # Generate translations
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("bho_Deva"),
            max_length=256,
            num_beams=5,
        )

        # Decode all chunks
        translated_chunks = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return ''.join(c for c in translated_chunks if c.strip()) if translated_chunks else None

    except Exception as e:
        logger.warning(f"Long text translation failed: {e}")
        return None


class HindiToBhojpuriTranslator:
    """Resumable Hindi→Bhojpuri translator with state tracking (NLLB-200 + CTranslate2)."""

    def __init__(self, data_dir=None, batch_size=50, max_consecutive_failures=10, cooldown_seconds=300):
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_consecutive_failures = max_consecutive_failures
        self.cooldown_seconds = cooldown_seconds
        self.state_file = self.data_dir / "translation_state.json"
        self.state = self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load translation state: {e}, starting fresh")
        return {
            "translated_hashes": {},
            "processed_raw_files": {},
            "next_output_batch_id": 0,
            "stats": {
                "total_records_translated": 0,
                "total_records_failed": 0,
                "total_api_calls": 0,
                "consecutive_failures": 0,
                "cooldowns_triggered": 0
            },
            "last_run_started_at": None,
            "last_run_ended_at": None,
            "last_run_stop_reason": None
        }

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _is_text_translated(self, text_hash):
        return text_hash in self.state["translated_hashes"]

    def _mark_text_translated(self, text_hash):
        self.state["translated_hashes"][text_hash] = True

    def _get_next_batch_id(self):
        batch_id = self.state["next_output_batch_id"]
        self.state["next_output_batch_id"] += 1
        return batch_id

    def _save_translated_batch(self, batch_data):
        batch_id = self._get_next_batch_id()
        output_file = self.data_dir / "hindi_translated_raw" / f"hindi_translated_batch_{batch_id:06d}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in batch_data:
                # Convert to cleaner format: only 'text' field expected by BhojpuriDataCleaner
                output_item = {
                    'text': item['bhojpuri_text'],
                    'source_hindi': item['hindi_text'],
                    'source': item['source']
                }
                json.dump(output_item, f, ensure_ascii=False)
                f.write('\n')

        return batch_id

    def run(self, source_raw_dirs: List[str], target_new_translations=100):
        """Main translation runner."""
        logger.info(f"Starting translation run (target: {target_new_translations} new translations)")
        import time
        self.state["last_run_started_at"] = time.strftime('%Y-%m-%dT%H:%M:%S')

        try:
            current_batch = []
            translations_done = 0
            files_processed = 0

            for source_dir_str in source_raw_dirs:
                source_dir = Path(source_dir_str)
                if not source_dir.exists():
                    logger.warning(f"Source directory not found: {source_dir}")
                    continue

                jsonl_files = sorted(source_dir.glob("*.jsonl"))
                logger.info(f"Processing {len(jsonl_files)} files from {source_dir}")

                for jsonl_file in jsonl_files:
                    file_key = f"{source_dir.name}/{jsonl_file.name}"

                    if file_key in self.state["processed_raw_files"]:
                        logger.debug(f"Skipping already-processed file: {file_key}")
                        continue

                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if translations_done >= target_new_translations:
                                    break

                                try:
                                    record = json.loads(line)
                                    hindi_text = record.get('text', '').strip()

                                    if not hindi_text:
                                        continue

                                    text_hash = md5(hindi_text.encode('utf-8')).hexdigest()

                                    if self._is_text_translated(text_hash):
                                        continue

                                    logger.info(f"Translating record {translations_done + 1}/{target_new_translations}...")
                                    translated = translate_long_text(hindi_text)

                                    if translated:
                                        self._mark_text_translated(text_hash)
                                        current_batch.append({
                                            'hindi_text': hindi_text,
                                            'bhojpuri_text': translated,
                                            'source': file_key,
                                            'translation_hash': text_hash
                                        })
                                        self.state["stats"]["total_records_translated"] += 1
                                        self.state["stats"]["total_api_calls"] += len(split_into_chunks(hindi_text))
                                        self.state["stats"]["consecutive_failures"] = 0
                                        translations_done += 1

                                        if len(current_batch) >= self.batch_size:
                                            self._save_translated_batch(current_batch)
                                            self._save_state()
                                            current_batch = []
                                    else:
                                        self.state["stats"]["total_records_failed"] += 1
                                        self.state["stats"]["consecutive_failures"] += 1

                                        if self.state["stats"]["consecutive_failures"] >= self.max_consecutive_failures:
                                            logger.error(f"Too many consecutive failures ({self.max_consecutive_failures}), cooling down...")
                                            self.state["stats"]["cooldowns_triggered"] += 1
                                            import time
                                            time.sleep(self.cooldown_seconds)
                                            self.state["stats"]["consecutive_failures"] = 0

                                except json.JSONDecodeError:
                                    logger.warning(f"JSON decode error at {jsonl_file}:{line_num}")
                                    continue

                                if translations_done >= target_new_translations:
                                    break

                        self.state["processed_raw_files"][file_key] = True
                        files_processed += 1

                    except Exception as e:
                        logger.error(f"Error processing file {jsonl_file}: {e}")
                        continue

                    if translations_done >= target_new_translations:
                        break

                if translations_done >= target_new_translations:
                    break

            # Save any remaining batch
            if current_batch:
                self._save_translated_batch(current_batch)

            self._save_state()
            import time
            self.state["last_run_stop_reason"] = "target_reached" if translations_done >= target_new_translations else "source_exhausted"
            self.state["last_run_ended_at"] = time.strftime('%Y-%m-%dT%H:%M:%S')
            self._save_state()

            logger.info(f"Translation run completed: {translations_done} translations, {files_processed} files processed")
            return {
                "translations_done": translations_done,
                "files_processed": files_processed,
                "stats": self.state["stats"]
            }

        except KeyboardInterrupt:
            logger.info("Translation run interrupted by user")
            self._save_state()
            raise
        except Exception as e:
            logger.error(f"Translation run failed: {e}")
            self._save_state()
            raise

    def self_test(self):
        """Test with 5 fixed Hindi sample sentences."""
        samples = [
            "यह एक परीक्षण है।",
            "नमस्ते, आपका स्वागत है।",
            "मेरा नाम क्या है?",
            "भारत एक सुंदर देश है।",
            "मुझे खाना पसंद है।"
        ]

        logger.info("Running self-test with 5 Hindi samples (NLLB-200)...")
        for i, sample in enumerate(samples, 1):
            translated = translate_long_text(sample)
            logger.info(f"Sample {i}:")
            logger.info(f"  Hindi: {sample}")
            logger.info(f"  Bhojpuri: {translated if translated else '(translation failed)'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hindi to Bhojpuri Translator (NLLB-200 local)")
    parser.add_argument("--source-dirs", nargs="+", default=[], help="Source directories with JSONL files")
    parser.add_argument("--target-new-translations", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run self-test and exit")
    args = parser.parse_args()

    translator = HindiToBhojpuriTranslator(data_dir=args.data_dir)

    if args.self_test:
        translator.self_test()
        return

    if not args.source_dirs:
        logger.error("No source directories specified. Use --source-dirs")
        return

    translator.run(source_raw_dirs=args.source_dirs, target_new_translations=args.target_new_translations)


if __name__ == "__main__":
    main()
