#!/usr/bin/env python3
"""
Thin wrapper to download Hindi books from archive.org (raw, unprocessed).
Does NOT clean/dedup/merge — that happens in the translation pipeline.
"""
import logging
from pathlib import Path
from .archive_org_downloader import BhojpuriArchiveOrgDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_hindi_books(data_dir=None, target_items=100, max_runtime_seconds=None):
    """
    Download Hindi books from archive.org (raw).
    """
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data")

    data_dir = Path(data_dir)

    downloader = BhojpuriArchiveOrgDownloader(
        language_query="Hindi",
        script_range=(0x0900, 0x097F),
        output_dir=str(data_dir / "hindi_books_raw"),
        state_file=str(data_dir / "hindi_books_archive_org_state.json"),
        batch_prefix="hindi_books"
    )

    result = downloader.run(target_new_items=target_items, max_runtime_seconds=max_runtime_seconds)
    logger.info(f"Hindi books download result: {result}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download Hindi books from archive.org (raw)")
    parser.add_argument("--target-items", type=int, default=100)
    parser.add_argument("--max-runtime", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    download_hindi_books(
        data_dir=args.data_dir,
        target_items=args.target_items,
        max_runtime_seconds=args.max_runtime
    )


if __name__ == "__main__":
    main()
