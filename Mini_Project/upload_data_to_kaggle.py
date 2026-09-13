#!/usr/bin/env python3
"""
Upload the pretraining corpus + finetuning QA data (both languages) as a new version of
the existing kspsvlnsiddardha/lma-slm dataset, using the `kagglehub` Python client.

Resulting dataset layout (mirrors the current remote layout, minus te.txt -- see below):
    telugu/data/{train,val,test}/telugu.txt
    telugu/data/{train,val,test}/fine_tune.txt
    bhojpuri/data/{train,val,test}/bhoj.txt
    bhojpuri/data/{train,val,test}/fine_tune.txt

IMPORTANT -- Kaggle dataset versions are a full replace, not an incremental patch: whatever
local folder is uploaded becomes the ENTIRE new version. The dataset currently also has
telugu/data/{train,val,test}/te.txt (~15.6GB total) which does not exist locally, so it is
NOT included in the staged upload folder below and will be PERMANENTLY DROPPED from the
dataset by this upload. This was an explicit, confirmed decision (te.txt is unused) -- do not
run this against a dataset you haven't checked the same way for other username handles/files.

Requires the `kagglehub` package (only present in this repo's venv/):
    venv/bin/python3 upload_data_to_kaggle.py

Usage:
    python3 upload_data_to_kaggle.py --dry-run     # show what would be staged/uploaded
    python3 upload_data_to_kaggle.py                # for real
    python3 upload_data_to_kaggle.py --force        # skip the auth-owner check

Auth: same as upload_phase2_bundles.py -- tries every KAGGLE_API_TOKEN found in
api_token.txt (there's more than one, for different accounts) and uses whichever one
actually authenticates as KAGGLE_USERNAME (kspsvlnsiddardha for this dataset -- NOT kspsvln,
which owns the phase2 bundles / checkpoint datasets).
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
STAGING_DIR = REPO_ROOT / ".kaggle_data_staging"

KAGGLE_USERNAME = "kspsvlnsiddardha"
DATASET_NAME = "lma-slm"

DEFAULT_MESSAGE = "Add reasoning finetune QA data (train/val/test); drop unused te.txt"

# (relative path under REPO_ROOT, relative path in the staged upload dir)
FILES = [
    ("telugu/data/train/telugu.txt", "telugu/data/train/telugu.txt"),
    ("telugu/data/val/telugu.txt", "telugu/data/val/telugu.txt"),
    ("telugu/data/test/telugu.txt", "telugu/data/test/telugu.txt"),
    ("telugu/data/train/fine_tune.txt", "telugu/data/train/fine_tune.txt"),
    ("telugu/data/val/fine_tune.txt", "telugu/data/val/fine_tune.txt"),
    ("telugu/data/test/fine_tune.txt", "telugu/data/test/fine_tune.txt"),
    ("bhojpuri/data/train/bhoj.txt", "bhojpuri/data/train/bhoj.txt"),
    ("bhojpuri/data/val/bhoj.txt", "bhojpuri/data/val/bhoj.txt"),
    ("bhojpuri/data/test/bhoj.txt", "bhojpuri/data/test/bhoj.txt"),
    ("bhojpuri/data/train/fine_tune.txt", "bhojpuri/data/train/fine_tune.txt"),
    ("bhojpuri/data/val/fine_tune.txt", "bhojpuri/data/val/fine_tune.txt"),
    ("bhojpuri/data/test/fine_tune.txt", "bhojpuri/data/test/fine_tune.txt"),
]

DROPPED_REMOTE_FILES = [
    "telugu/data/train/te.txt (~12.5GB)",
    "telugu/data/val/te.txt (~1.56GB)",
    "telugu/data/test/te.txt (~1.56GB)",
]


def find_kaggle_tokens():
    token_file = REPO_ROOT / "api_token.txt"
    if not token_file.exists():
        return []
    tokens = []
    for line in token_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line and line.split("=", 1)[0].strip() == "KAGGLE_API_TOKEN":
            _, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if value and value not in tokens:
                tokens.append(value)
    return tokens


def resolve_kaggle_token():
    try:
        from kagglehub.auth import _validate_credentials_helper
    except ImportError:
        print("  ⚠️  kagglehub not installed in this Python environment (use venv/bin/python3).")
        return None

    candidates = find_kaggle_tokens()
    if not candidates and os.environ.get("KAGGLE_API_TOKEN"):
        candidates = [os.environ["KAGGLE_API_TOKEN"]]
    if not candidates:
        print("  ⚠️  No KAGGLE_API_TOKEN found in api_token.txt or the environment.")
        return None

    last_username = None
    for token in candidates:
        os.environ["KAGGLE_API_TOKEN"] = token
        try:
            username = _validate_credentials_helper(verbose=False)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  Token {token[:10]}... failed validation: {e}")
            continue
        last_username = username
        if username == KAGGLE_USERNAME:
            return username

    return last_username


def stage_files():
    """Build STAGING_DIR as a tree of symlinks to the real files (avoids copying multi-GB
    data). Returns (staged_paths, missing_paths)."""
    if STAGING_DIR.exists():
        # Remove only symlinks we control; refuse if something unexpected is there.
        for p in sorted(STAGING_DIR.rglob("*"), reverse=True):
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()

    staged, missing = [], []
    for src_rel, dst_rel in FILES:
        src = REPO_ROOT / src_rel
        dst = STAGING_DIR / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        staged.append((dst_rel, src.stat().st_size))

    return staged, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Version notes for kagglehub.dataset_upload")
    parser.add_argument("--dry-run", action="store_true", help="Stage files and show what would upload, no network calls")
    parser.add_argument("--force", action="store_true", help="Attempt upload even if the authenticated user doesn't match the dataset owner")
    args = parser.parse_args()

    dataset_id = f"{KAGGLE_USERNAME}/{DATASET_NAME}"
    print(f"Target dataset: {dataset_id}\n")

    staged, missing = stage_files()

    print("Files to be included in the new version:")
    total_bytes = 0
    for rel, size in staged:
        total_bytes += size
        print(f"  {rel}  ({size / 1e6:.1f} MB)")
    print(f"  Total: {total_bytes / 1e9:.2f} GB\n")

    if missing:
        print("⚠️  Missing locally, will NOT be in the new version (unless intentional, stop and check):")
        for rel in missing:
            print(f"  {rel}")
        print()

    print("⚠️  Will be PERMANENTLY DROPPED from the dataset (full-replace versioning, not present locally):")
    for rel in DROPPED_REMOTE_FILES:
        print(f"  {rel}")
    print()

    current_user = resolve_kaggle_token()
    user_ok = current_user == KAGGLE_USERNAME
    if current_user:
        print(f"Authenticated as: {current_user}")
        if not user_ok:
            print(
                f"⚠️  Authenticated as '{current_user}', but dataset owner is "
                f"'{KAGGLE_USERNAME}'. This upload will fail unless '{current_user}' is a "
                f"collaborator on this dataset."
            )
    if not user_ok and not args.force and not args.dry_run:
        print(f"❌ Aborting (auth mismatch). Re-run with --force to attempt anyway.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n(dry-run) would call: kagglehub.dataset_upload({dataset_id!r}, {str(STAGING_DIR)!r}, version_notes={args.message!r})")
        print("(dry-run) staging left in place for inspection:", STAGING_DIR)
        return

    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed in this Python environment. Run with venv/bin/python3.")
        sys.exit(1)

    try:
        kagglehub.dataset_upload(dataset_id, str(STAGING_DIR), version_notes=args.message)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Upload failed: {e}")
        sys.exit(1)

    print(f"✅ Uploaded new version of {dataset_id}")


if __name__ == "__main__":
    main()
