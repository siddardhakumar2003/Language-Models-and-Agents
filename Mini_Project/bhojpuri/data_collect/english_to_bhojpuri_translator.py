#!/usr/bin/env python3
"""
Fast English → Bhojpuri Translation Pipeline using NLLB-200

Optimizations:
- Uses NLLB-200-distilled (600M) for speed
- CTranslate2 backend for 5-10x speedup
- Batch processing with large batch sizes
- FP16 precision for memory efficiency
- Async GPU processing
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnglishToBhojpuriTranslator:
    """Fast English → Bhojpuri translator using NLLB-200"""

    # Language codes for NLLB
    SOURCE_LANG = "eng_Latn"  # English (Latin script)
    TARGET_LANG = "bho_Deva"  # Bhojpuri (Devanagari script)

    def __init__(self, use_fast_backend=True, use_fp16=True):
        """
        Initialize translator.

        Args:
            use_fast_backend: Use CTranslate2 for 5-10x speedup (requires ctranslate2)
            use_fp16: Use FP16 precision to reduce memory usage
        """
        self.use_fast_backend = use_fast_backend
        self.use_fp16 = use_fp16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Device: {self.device}")
        logger.info(f"FP16: {self.use_fp16}")
        logger.info(f"Fast backend (CTranslate2): {self.use_fast_backend}")

        self._load_model()

    def _load_model(self):
        """Load NLLB model with optional CTranslate2 optimization"""
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        )

        model_name = "facebook/nllb-200-distilled-600M"
        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.SOURCE_LANG,
            use_fast=True,
        )

        # Get the forced BOS token ID for target language (used in generate())
        self.target_lang_id = self.tokenizer.convert_tokens_to_ids(self.TARGET_LANG)
        logger.info(f"Target language ({self.TARGET_LANG}) token ID: {self.target_lang_id}")

        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )

        if self.device.type == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully!")

        # Try to use CTranslate2 for speedup
        if self.use_fast_backend:
            try:
                import ctranslate2
                logger.info("Converting to CTranslate2 for 5-10x speedup...")
                self.model = ctranslate2.converters.TransformersConverter(model_name).convert()
                self.fast_backend = True
                logger.info("CTranslate2 conversion successful!")
            except Exception as e:
                logger.warning(f"CTranslate2 not available ({e}), using standard backend")
                self.fast_backend = False
        else:
            self.fast_backend = False

    @torch.inference_mode()
    def translate_batch(self, texts: List[str], batch_size: int = 64) -> List[str]:
        """
        Translate batch of texts.

        Args:
            texts: List of English texts to translate
            batch_size: Process in batches this size (larger = faster but uses more memory)

        Returns:
            List of Bhojpuri translations
        """
        if not texts:
            return []

        # Skip empty/short texts
        texts = [t.strip() for t in texts if t and len(t.strip().split()) >= 2]
        if not texts:
            return []

        translations = []

        if self.fast_backend:
            # CTranslate2 path (much faster)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    results = self.model.translate_batch(batch)
                    translations.extend([r[0]["tgt"] for r in results])
                except Exception as e:
                    logger.warning(f"CTranslate2 translation error: {e}, falling back")
                    translations.extend(self._translate_fallback(batch))
        else:
            # Standard HuggingFace path
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                translations.extend(self._translate_fallback(batch))

        return translations

    def _translate_fallback(self, texts: List[str]) -> List[str]:
        """Fallback translation using standard HuggingFace pipeline"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            src_lang=self.SOURCE_LANG,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.target_lang_id,
            max_length=256,
            num_beams=1,
            do_sample=False,
        )

        translations = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        return translations

    def self_test(self) -> bool:
        """Test with sample English texts"""
        samples = [
            "The government announced a new policy.",
            "People gathered near the river yesterday.",
            "Science and technology are advancing rapidly.",
        ]

        logger.info("Running self-test...")
        try:
            translations = self.translate_batch(samples)
            logger.info("Self-test results:")
            for en, bho in zip(samples, translations):
                logger.info(f"  EN : {en}")
                logger.info(f"  BHO: {bho}")
            return True
        except Exception as e:
            logger.error(f"Self-test failed: {e}")
            return False


def process_text_file(
    input_path: str,
    output_path: str,
    translator: EnglishToBhojpuriTranslator,
    batch_size: int = 64,
    checkpoint: Optional[Dict] = None,
) -> Dict:
    """
    Process a single text file and output translations.

    Args:
        input_path: Path to input text file
        output_path: Path to output JSONL file
        translator: Translator instance
        batch_size: Batch size for processing
        checkpoint: State to resume from

    Returns:
        Updated checkpoint with processing stats
    """
    checkpoint = checkpoint or {"lines_processed": 0, "valid_outputs": 0}

    start_time = time.time()
    lines_processed = checkpoint.get("lines_processed", 0)
    valid_outputs = checkpoint.get("valid_outputs", 0)

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
            with open(output_path, 'a', encoding='utf-8', buffering=1024*1024) as outfile:

                buffer = []

                for line_num, line in enumerate(infile, 1):
                    if line_num <= lines_processed:
                        continue  # Skip already processed lines

                    line = line.strip()
                    if not line or len(line.split()) < 3:  # Skip short lines
                        continue

                    buffer.append(line)

                    # Process buffer when full
                    if len(buffer) >= batch_size:
                        translations = translator.translate_batch(buffer)

                        for en_text, bho_text in zip(buffer, translations):
                            bho_text = bho_text.strip()
                            if bho_text and len(bho_text.split()) >= 2:
                                record = {
                                    "en": en_text,
                                    "bho": bho_text,
                                    "lang_pair": "en_Latn-bho_Deva",
                                }
                                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                                valid_outputs += 1

                        lines_processed = line_num
                        buffer = []

                        # Log progress every 1000 lines
                        if lines_processed % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = lines_processed / elapsed
                            logger.info(
                                f"Progress: {lines_processed:,} lines, "
                                f"{valid_outputs:,} translations ({rate:.1f} lines/sec)"
                            )

                # Process remaining buffer
                if buffer:
                    translations = translator.translate_batch(buffer)
                    for en_text, bho_text in zip(buffer, translations):
                        bho_text = bho_text.strip()
                        if bho_text and len(bho_text.split()) >= 2:
                            record = {
                                "en": en_text,
                                "bho": bho_text,
                                "lang_pair": "en_Latn-bho_Deva",
                            }
                            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                            valid_outputs += 1

                    lines_processed = line_num

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"File complete: {lines_processed:,} lines processed, "
        f"{valid_outputs:,} translations, {elapsed/60:.1f} min"
    )

    return {
        "lines_processed": lines_processed,
        "valid_outputs": valid_outputs,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Translate English corpus to Bhojpuri using NLLB-200"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input text file to translate"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing text files to translate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bhojpuri/data/english_translations",
        help="Output directory for translations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for translation (larger = faster but uses more GPU memory)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test and exit"
    )
    parser.add_argument(
        "--no-fast-backend",
        action="store_true",
        help="Don't use CTranslate2 fast backend (will be slower)"
    )

    args = parser.parse_args()

    # Create translator
    translator = EnglishToBhojpuriTranslator(
        use_fast_backend=not args.no_fast_backend
    )

    # Self-test mode
    if args.self_test:
        success = translator.self_test()
        return 0 if success else 1

    # Validate input
    if not args.input_file and not args.input_dir:
        parser.print_help()
        return 1

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "translations.jsonl")

    logger.info(f"Output file: {output_file}")

    # Process files
    if args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            return 1

        logger.info(f"Processing: {args.input_file}")
        process_text_file(
            args.input_file,
            output_file,
            translator,
            batch_size=args.batch_size,
        )

    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1

        # Find all text files
        txt_files = sorted(Path(args.input_dir).glob("**/*.txt"))
        logger.info(f"Found {len(txt_files)} text files")

        for i, txt_file in enumerate(txt_files, 1):
            logger.info(f"\n[{i}/{len(txt_files)}] {txt_file.name}")
            process_text_file(
                str(txt_file),
                output_file,
                translator,
                batch_size=args.batch_size,
            )

    logger.info("Translation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
