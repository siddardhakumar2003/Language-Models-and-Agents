#!/usr/bin/env python3
"""
Merge English→Bhojpuri (fineweb-edu via NLLB-200, Kaggle) translations into
the Bhojpuri training corpus. One-shot batch merge of an already-completed
translation job output at data/Translated/bhojpuri_translations/translations.jsonl.
"""
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .data_cleaner import BhojpuriDataCleaner
from .download_hf_corpus import deduplicate_with_existing_corpus
from .ocr_merge import merge_ocr_into_splits
from .update_config_ocr_fixed import update_config_bhojpuri

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RAW_DIR_NAME = "english_translated_raw"
CLEANED_DIR_NAME = "english_translated_cleaned"
TEMP_BATCH_FILENAME = "english_translated_new_batch.txt"
SOURCE_JSONL = "Translated/bhojpuri_translations/translations.jsonl"


def reshape_translations_to_raw(data_dir: Path, source_jsonl: Path, raw_dir: Path) -> int:
    """
    Stream translations.jsonl, extract 'bho' field only, write reshaped
    single JSONL file into raw_dir in {"text": ...} format expected by
    BhojpuriDataCleaner. Skips 'en' and 'lang_pair' fields entirely.
    Streams line-by-line (no full-file load) since input is 321MB/425K lines.
    Returns count of records written.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_file = raw_dir / "english_translated_batch_000000.jsonl"

    written = 0
    skipped_empty = 0

    logger.info(f"Reshaping {source_jsonl.name} to {output_file.name}")

    with open(source_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_num % 10000 == 0:
                    logger.warning(f"Skipped unparseable JSON at line {line_num}")
                continue

            bho_text = record.get('bho', '').strip()
            if not bho_text:
                skipped_empty += 1
                continue

            json.dump({'text': bho_text}, fout, ensure_ascii=False)
            fout.write('\n')
            written += 1

    logger.info(
        f"Reshaped to {output_file.name}: "
        f"{written} records written, {skipped_empty} empty 'bho' fields skipped"
    )
    return written


def clean_translations(data_dir: Path) -> int:
    """
    Run BhojpuriDataCleaner on the reshaped raw dir.
    Returns total_kept from cleaning_report.json.
    """
    logger.info("Starting cleaning pipeline...")

    cleaner = BhojpuriDataCleaner(
        input_dir=str(data_dir / RAW_DIR_NAME),
        output_dir=str(data_dir / CLEANED_DIR_NAME),
    )
    cleaner.run_cleaning()

    report_file = data_dir / CLEANED_DIR_NAME / "cleaning_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
            total_kept = report.get('total_kept', 0)
            logger.info(f"Cleaning complete: {total_kept} kept, {report.get('total_rejected', 0)} rejected")
            return total_kept

    logger.warning("Cleaning report not found; assuming 0 records kept")
    return 0


def dedupe_against_corpus(data_dir: Path) -> None:
    """
    Deduplicate cleaned batch against existing corpus, using the same
    temp-dir-swap trick as run_phase3_translation.py to avoid source==output-dir
    I/O issues in deduplicate_with_existing_corpus.
    """
    logger.info("Deduplicating against existing corpus...")

    with tempfile.TemporaryDirectory() as tmp:
        temp_dedup = Path(tmp) / "english_translated_dedup"
        temp_dedup.mkdir()

        deduplicate_with_existing_corpus(
            data_dir=data_dir,
            source_cleaned_dir=data_dir / CLEANED_DIR_NAME,
            output_cleaned_dir=temp_dedup,
        )

        # Swap the temp result back over the original cleaned dir
        shutil.rmtree(data_dir / CLEANED_DIR_NAME)
        shutil.move(str(temp_dedup), str(data_dir / CLEANED_DIR_NAME))

    logger.info("Deduplication complete")


def merge_and_resplit(data_dir: Path) -> dict:
    """
    Merge cleaned+deduped batch into train/val/test via merge_ocr_into_splits.
    """
    logger.info("Merging into corpus and re-splitting...")

    result = merge_ocr_into_splits(
        data_dir,
        accumulator_filename="bhoj.txt",
        split_filename="bhoj.txt",
        cleaned_dir_name=CLEANED_DIR_NAME,
        temp_batch_filename=TEMP_BATCH_FILENAME,
    )

    logger.info("Merge and re-split complete")
    return result


def archive_cleaned_dir(data_dir: Path) -> None:
    """Archive cleaned directory to a timestamped name."""
    cleaned_dir = data_dir / CLEANED_DIR_NAME
    if cleaned_dir.exists() and list(cleaned_dir.glob("*")):
        archive_dir = data_dir / f"{CLEANED_DIR_NAME}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(cleaned_dir), str(archive_dir))
        logger.info(f"Archived {CLEANED_DIR_NAME} to {archive_dir.name}")


def archive_raw_dir(data_dir: Path) -> None:
    """Archive raw directory to a timestamped name."""
    raw_dir = data_dir / RAW_DIR_NAME
    if raw_dir.exists() and list(raw_dir.glob("*")):
        archive_dir = data_dir / f"{RAW_DIR_NAME}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(raw_dir), str(archive_dir))
        logger.info(f"Archived {RAW_DIR_NAME} to {archive_dir.name}")


def patch_config(data_dir: Path, raw_count: int, cleaned_kept: int, merge_result: dict) -> None:
    """
    Manual config.json patches beyond what update_config_bhojpuri covers:
    - config["phase3_fineweb_translation"]: overwrite stale BART/T.ipynb
      description with real NLLB-200 stats, status -> COMPLETE
    - config["project_status"]["phase3_fineweb_translation"]: "READY" -> "COMPLETE"
    - config["total_training_lines"]: recompute from actual split sums
    - config["data_composition"]["total_lines"] and size_mb: recompute
    - config["data_sources"]: append new entry if not present
    """
    logger.info("Patching config.json...")

    config_path = data_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Recompute exact line counts from split files
    train_lines = sum(1 for _ in open(data_dir / "train" / "bhoj.txt", encoding='utf-8'))
    val_lines = sum(1 for _ in open(data_dir / "val" / "bhoj.txt", encoding='utf-8'))
    test_lines = sum(1 for _ in open(data_dir / "test" / "bhoj.txt", encoding='utf-8'))
    total_lines = train_lines + val_lines + test_lines

    total_bytes = sum(
        (data_dir / s / "bhoj.txt").stat().st_size
        for s in ('train', 'val', 'test')
    )

    # Update phase3_fineweb_translation block
    config["phase3_fineweb_translation"] = {
        "source": "fineweb-edu dataset (Kaggle: nameonlu/fineweb-edu)",
        "translation_method": "NLLB-200 (english_to_bhojpuri_translator.py, en_Latn→bho_Deva, GPU inference on Kaggle)",
        "raw_records_translated": raw_count,
        "cleaned_lines_kept": cleaned_kept,
        "merged_lines": merge_result.get('total_lines', 0) - merge_result.get('accumulator_before', 0),
        "last_run_at": datetime.now().isoformat(),
        "status": "COMPLETE",
    }

    # Update project_status
    config["project_status"]["phase3_fineweb_translation"] = "COMPLETE"
    config["project_status"]["last_updated"] = datetime.now().isoformat()

    # Fix total_training_lines (also corrects pre-existing 776801 vs 776810 drift)
    config["total_training_lines"] = total_lines
    config["data_composition"]["total_lines"] = total_lines
    config["data_composition"]["size_mb"] = round(total_bytes / 1024 / 1024, 1)

    # Append data_sources entry if not present
    source_str = "English→Bhojpuri Machine Translation (NLLB-200, Kaggle GPU; source: fineweb-edu)"
    if "data_sources" not in config:
        config["data_sources"] = []
    if source_str not in config["data_sources"]:
        config["data_sources"].append(source_str)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Config patched: total_training_lines={total_lines}, "
        f"token_progress updated by update_config_bhojpuri"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge English→Bhojpuri fineweb-edu translations into corpus"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to bhojpuri/data directory (default: auto-detect)"
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=None,
        help="Path to translations.jsonl (default: data/Translated/bhojpuri_translations/translations.jsonl)"
    )

    args = parser.parse_args()

    # Auto-detect data_dir if not provided
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        args.data_dir = script_dir.parent / "data"

    data_dir = args.data_dir
    source_jsonl = args.source_jsonl or (data_dir / SOURCE_JSONL)

    # Validate inputs
    if not source_jsonl.exists():
        logger.error(f"Source file not found: {source_jsonl}")
        return 1

    if not (data_dir / "config.json").exists():
        logger.error(f"Config not found: {data_dir / 'config.json'}")
        return 1

    logger.info("=" * 70)
    logger.info("ENGLISH → BHOJPURI (fineweb-edu) TRANSLATION MERGE")
    logger.info("=" * 70)
    logger.info(f"Source: {source_jsonl}")
    logger.info(f"Data dir: {data_dir}")

    # Step 1: Reshape
    raw_count = reshape_translations_to_raw(data_dir, source_jsonl, data_dir / RAW_DIR_NAME)
    if raw_count == 0:
        logger.error("No records to merge, aborting")
        return 1

    # Step 2: Clean
    cleaned_kept = clean_translations(data_dir)

    # Step 3: Dedupe
    dedupe_against_corpus(data_dir)

    # Step 4: Merge and re-split
    merge_result = merge_and_resplit(data_dir)
    if not merge_result:
        logger.error("Merge failed, aborting")
        return 1

    # Step 5: Archive
    archive_cleaned_dir(data_dir)
    archive_raw_dir(data_dir)

    # Step 6: Update config (fixed version)
    update_config_bhojpuri(str(data_dir / "config.json"), str(data_dir))

    # Step 7: Patch config (manual additions)
    patch_config(data_dir, raw_count, cleaned_kept, merge_result)

    logger.info("=" * 70)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Raw records:      {raw_count:,}")
    logger.info(f"  Cleaned kept:     {cleaned_kept:,}")
    logger.info(f"  Merged lines:     {merge_result.get('total_lines', 0) - merge_result.get('accumulator_before', 0):,}")
    logger.info(f"  Accumulator:      {merge_result.get('accumulator_before', 0):,} → {merge_result.get('accumulator_after', 0):,}")
    logger.info(f"  Train/Val/Test:   {merge_result.get('train_lines', 0):,} / {merge_result.get('val_lines', 0):,} / {merge_result.get('test_lines', 0):,}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
