#!/usr/bin/env python3
import json, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_corpus_tokens_estimate(data_dir, language="telugu"):
    total_bytes = 0
    files_to_check = ["te.txt", "telugu.txt"] if language == "telugu" else ["bhoj.txt"]
    for fname in files_to_check:
        fpath = data_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            total_bytes += size
            logger.info(f"  {fname}: {size / 1024 / 1024:.1f} MB")
    return total_bytes // 4, total_bytes

def recompute_splits_block(data_dir, language="telugu"):
    splits = {"train": {}, "val": {}, "test": {}}
    split_filenames = {"train": "telugu.txt", "val": "telugu.txt", "test": "telugu.txt"} if language == "telugu" else {"train": "bhoj.txt", "val": "bhoj.txt", "test": "bhoj.txt"}
    for split_name, filename in split_filenames.items():
        split_file = data_dir / split_name / filename
        if split_file.exists():
            try:
                line_count = sum(1 for _ in open(split_file))
                splits[split_name]["lines"] = line_count
                splits[split_name]["percentage"] = 0
                logger.info(f"  {split_name}/{filename}: {line_count} lines")
            except Exception as e:
                logger.warning(f"Could not count {split_file}: {e}")
                splits[split_name]["lines"] = 0
    return splits

def get_cumulative_ocr_stats(data_dir):
    stats_file = data_dir / "ocr_cumulative_stats.json"
    if not stats_file.exists():
        return {"total_files_processed": 0, "total_pages_extracted": 0, "total_lines_kept": 0, "total_lines_rejected": 0}
    try:
        with open(stats_file, 'r') as f:
            return json.load(f)
    except:
        return {"total_files_processed": 0, "total_pages_extracted": 0, "total_lines_kept": 0, "total_lines_rejected": 0}

def update_config_telugu(config_path, data_dir):
    config_path = Path(config_path)
    data_dir = Path(data_dir)
    logger.info(f"Updating {config_path}")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    logger.info("Computing token progress...")
    total_tokens, total_bytes = compute_corpus_tokens_estimate(data_dir, language="telugu")
    logger.info("Reading OCR cumulative stats...")
    cumulative_stats = get_cumulative_ocr_stats(data_dir)
    logger.info("Recomputing splits...")
    splits = recompute_splits_block(data_dir, language="telugu")
    config["ocr_data"] = {"source": "archive.org pre-OCR'd text (djvu.txt) + news crawling",
                         "files_processed": cumulative_stats.get("total_files_processed", 0),
                         "pages_extracted": cumulative_stats.get("total_pages_extracted", 0),
                         "lines_kept_after_cleaning": cumulative_stats.get("total_lines_kept", 0)}
    target_tokens = config.get("target_tokens", 500_000_000)
    progress_percent = 100.0 * total_tokens / target_tokens if target_tokens > 0 else 0
    config["token_progress"] = {"total_corpus_tokens_estimate": total_tokens, "total_corpus_bytes": total_bytes,
                               "target_tokens": target_tokens, "progress_percent": progress_percent,
                               "progress_str": f"{total_tokens/1e6:.1f}M / {target_tokens/1e6:.0f}M ({progress_percent:.1f}%)"}
    config["splits"] = splits
    config["updated_at"] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"✓ Config updated: {total_tokens/1e6:.1f}M / {target_tokens/1e6:.0f}M tokens ({progress_percent:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update Telugu config.json (fixed)")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parent.parent / "data"
    update_config_telugu(str(args.data_dir / "config.json"), str(args.data_dir))

if __name__ == "__main__":
    main()
