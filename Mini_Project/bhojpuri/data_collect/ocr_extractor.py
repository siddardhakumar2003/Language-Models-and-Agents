#!/usr/bin/env python3
"""
OCR-based text extraction from scanned books/news images for Bhojpuri.

Walks a source directory of PDFs and images, extracts text via Tesseract (Hindi/Devanagari proxy),
applies OCR-specific pre-cleaning, and outputs JSONL batches compatible
with the existing BhojpuriDataCleaner.

Usage:
    python3 ocr_extractor.py [--source-dir SRCDIR] [--output-dir OUTDIR]
"""

import json
import logging
import os
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Generator, List, Dict, Tuple
from PIL import Image
import pytesseract

try:
    import fitz
except ImportError:
    fitz = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BhojpuriOCRExtractor:
    def __init__(
        self,
        source_dir: str = None,
        output_dir: str = None,
        state_file: str = None,
        lang: str = "hin",
        dpi: int = 300,
    ):
        if source_dir is None:
            source_dir = str(Path(__file__).resolve().parent / "ocr_sources")
        if output_dir is None:
            output_dir = str(Path(__file__).resolve().parent.parent / "data" / "ocr_raw")
        if state_file is None:
            state_file = str(Path(__file__).resolve().parent.parent / "data" / "ocr_state.json")

        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.state_file = Path(state_file)
        self.lang = lang
        self.dpi = dpi

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processed_files = {}
        self.next_batch_id = 0
        self.token_count = 0

        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.processed_files = state.get('processed_files', {})
                    self.next_batch_id = state.get('next_batch_id', 0)
                    self.token_count = state.get('token_count', 0)
                    logger.info(
                        f"Loaded state: {len(self.processed_files)} processed files, "
                        f"next_batch_id={self.next_batch_id}, token_count={self.token_count}"
                    )
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                self.processed_files = {}
                self.next_batch_id = 0
                self.token_count = 0

    def _save_state(self) -> None:
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_files': self.processed_files,
                    'next_batch_id': self.next_batch_id,
                    'token_count': self.token_count,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _discover_source_files(self) -> List[Path]:
        if not self.source_dir.exists():
            logger.warning(f"Source directory not found: {self.source_dir}")
            return []

        supported_exts = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}
        files = [
            f for f in self.source_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in supported_exts
        ]
        return sorted(files)

    def _extract_pdf_pages(self, pdf_path: Path) -> Generator[Tuple[int, Image.Image], None, None]:
        if fitz is None:
            logger.error("PyMuPDF (fitz) not installed. Cannot process PDFs.")
            return

        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                zoom_matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=zoom_matrix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                yield page_num, img
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")

    def _ocr_image(self, image: Image.Image) -> str:
        try:
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text
        except Exception as e:
            logger.error(f"Error running OCR: {e}")
            return ""

    def _strip_running_headers_footers(self, pages: List[str]) -> List[str]:
        """Remove repeated headers/footers appearing in >=30% of pages (min 3)."""
        if len(pages) < 3:
            return pages

        first_lines = {}
        last_lines = {}

        for page_text in pages:
            lines = page_text.split('\n')
            if lines:
                first = lines[0].strip()
                last = lines[-1].strip()
                if first:
                    first_lines[first] = first_lines.get(first, 0) + 1
                if last:
                    last_lines[last] = last_lines.get(last, 0) + 1

        threshold = max(3, int(len(pages) * 0.30))
        header_lines = {line for line, count in first_lines.items() if count >= threshold}
        footer_lines = {line for line, count in last_lines.items() if count >= threshold}

        result = []
        for page_text in pages:
            lines = page_text.split('\n')
            filtered = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                is_header = (i == 0 and stripped in header_lines)
                is_footer = (i == len(lines) - 1 and stripped in footer_lines)
                if not (is_header or is_footer):
                    filtered.append(line)
            result.append('\n'.join(filtered))

        return result

    def _preclean_ocr_text(self, pages: List[str]) -> List[str]:
        """Document-level pre-clean before per-page cleanup."""
        pages = self._strip_running_headers_footers(pages)

        cleaned = []
        for page_text in pages:
            lines = page_text.split('\n')
            lines = [ln for ln in lines if len(ln.strip()) > 2 or ln.strip().isdigit()]
            page_text = '\n'.join(lines)

            page_text = re.sub(r'(\w)-\n(\w)', r'\1\2', page_text)
            page_text = re.sub(r'\s*\n\s*', ' ', page_text, flags=re.UNICODE)
            page_text = page_text.strip()

            if page_text:
                cleaned.append(page_text)

        return cleaned

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _should_skip_file(self, file_path: Path) -> bool:
        rel_path = file_path.relative_to(self.source_dir).as_posix()
        if rel_path not in self.processed_files:
            return False

        stat = file_path.stat()
        recorded = self.processed_files[rel_path]
        if stat.st_size != recorded.get('size_bytes') or stat.st_mtime != recorded.get('mtime'):
            return False

        return True

    def _compute_sha256(self, file_path: Path) -> str:
        try:
            sha = sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception as e:
            logger.error(f"Error computing SHA256 for {file_path}: {e}")
            return ""

    def _save_batch(self, records: List[Dict], batch_id: int) -> None:
        output_file = self.output_dir / f"batch_{batch_id:06d}.jsonl"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"Saved batch {batch_id} with {len(records)} records to {output_file}")
        except Exception as e:
            logger.error(f"Error saving batch {batch_id}: {e}")

    def run_extraction(self, batch_size: int = 100) -> Dict:
        logger.info("Starting OCR extraction (Bhojpuri, Hindi proxy)...")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")

        source_files = self._discover_source_files()
        if not source_files:
            logger.warning("No source files found.")
            return {'total_files': 0, 'total_pages': 0, 'total_tokens': 0}

        total_files = 0
        total_pages = 0
        current_batch = []
        batch_token_count = 0

        for file_path in source_files:
            if self._should_skip_file(file_path):
                logger.info(f"Skipping already-processed: {file_path.name}")
                continue

            logger.info(f"Processing {file_path.name}...")
            pages = []
            pages_extracted = 0

            if file_path.suffix.lower() == '.pdf':
                for page_num, image in self._extract_pdf_pages(file_path):
                    ocr_text = self._ocr_image(image)
                    if ocr_text:
                        pages.append(ocr_text)
                        pages_extracted += 1
            else:
                try:
                    image = Image.open(file_path)
                    ocr_text = self._ocr_image(image)
                    if ocr_text:
                        pages.append(ocr_text)
                        pages_extracted = 1
                except Exception as e:
                    logger.error(f"Error processing image {file_path}: {e}")
                    continue

            if pages:
                pages = self._preclean_ocr_text(pages)

                for page_text in pages:
                    if page_text:
                        tokens = self._estimate_tokens(page_text)
                        record = {
                            'text': page_text,
                            'tokens': tokens,
                            'timestamp': datetime.now().isoformat(),
                        }
                        current_batch.append(record)
                        batch_token_count += tokens
                        total_pages += 1

                        if len(current_batch) >= batch_size:
                            self._save_batch(current_batch, self.next_batch_id)
                            self.token_count += batch_token_count
                            self.next_batch_id += 1
                            self._save_state()

                            current_batch = []
                            batch_token_count = 0

                rel_path = file_path.relative_to(self.source_dir).as_posix()
                self.processed_files[rel_path] = {
                    'mtime': file_path.stat().st_mtime,
                    'size_bytes': file_path.stat().st_size,
                    'pages_extracted': pages_extracted,
                    'sha256': self._compute_sha256(file_path),
                }
                self._save_state()
                total_files += 1

        if current_batch:
            self._save_batch(current_batch, self.next_batch_id)
            self.token_count += batch_token_count
            self.next_batch_id += 1
            self._save_state()

        logger.info(f"Extraction complete: {total_files} files, {total_pages} pages, {self.token_count} tokens")
        return {
            'total_files': total_files,
            'total_pages': total_pages,
            'total_tokens': self.token_count,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from scanned documents via OCR (Bhojpuri, Hindi proxy)")
    parser.add_argument(
        '--source-dir',
        help='Directory containing PDFs/images (default: bhojpuri/data_collect/ocr_sources/)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for JSONL batches (default: bhojpuri/data/ocr_raw/)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for saving (default: 100)'
    )

    args = parser.parse_args()

    extractor = BhojpuriOCRExtractor(source_dir=args.source_dir, output_dir=args.output_dir)
    result = extractor.run_extraction(batch_size=args.batch_size)

    print("\n" + "=" * 70)
    print(f"Total files processed: {result['total_files']}")
    print(f"Total pages extracted: {result['total_pages']}")
    print(f"Total tokens: {result['total_tokens']}")
    print("=" * 70 + "\n")

    return 0 if result['total_files'] > 0 or result['total_pages'] == 0 else 1


if __name__ == "__main__":
    exit(main())
