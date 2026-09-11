#!/usr/bin/env python3
"""
Upload the Phase 2 training bundles (train/kaggle_bundle/) to their existing
Kaggle datasets as new versions, using the `kagglehub` Python client.

Each bundle contains configs/, model/, tokenizer/, train/ (code + configs,
no raw corpus).

Requires the `kagglehub` package (only present in this repo's venv/):
    venv/bin/python3 upload_phase2_bundles.py

Usage:
    python3 upload_phase2_bundles.py                       # both languages
    python3 upload_phase2_bundles.py --lang telugu
    python3 upload_phase2_bundles.py --lang bhojpuri --message "Custom note"
    python3 upload_phase2_bundles.py --dry-run              # show what would upload, no network calls
    python3 upload_phase2_bundles.py --force                # skip the auth-owner check

Auth: api_token.txt at the repo root holds tokens for more than one Kaggle
account (one KAGGLE_API_TOKEN export per line). This script tries every one
of them live against Kaggle and uses whichever actually authenticates as
KAGGLE_USERNAME, rather than guessing based on line order.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

KAGGLE_USERNAME = "kspsvln"

BUNDLES = {
    "telugu": {
        "bundle_dir": REPO_ROOT / "telugu" / "train" / "kaggle_bundle",
        "dataset_name": "lma-telugu-phase2",
    },
    "bhojpuri": {
        "bundle_dir": REPO_ROOT / "bhojpuri" / "train" / "kaggle_bundle",
        "dataset_name": "lma-bhojpuri-phase2",
    },
}

DEFAULT_MESSAGE = (
    "Resize for from-scratch retrain: Telugu 20K vocab/25.5M params, "
    "Bhojpuri 16K vocab/15.1M params, max_seq_length=256"
)


def find_kaggle_tokens():
    """Return every KAGGLE_API_TOKEN value found in api_token.txt, in file order.

    api_token.txt holds tokens for more than one Kaggle account. Rather than
    guess which line is "current" (first vs. last), we try every candidate
    live against Kaggle and pick whichever one actually authenticates as
    KAGGLE_USERNAME -- see resolve_kaggle_token().
    """
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
    """Try every candidate token and use the one that authenticates as KAGGLE_USERNAME.

    Sets KAGGLE_API_TOKEN in the environment to the winning token (or the last
    candidate tried, if none matched) and returns the resolved username (or
    None if nothing could be validated).
    """
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
        except Exception as e:  # noqa: BLE001 - keep trying other candidates
            print(f"  ⚠️  Token {token[:10]}... failed validation: {e}")
            continue
        last_username = username
        if username == KAGGLE_USERNAME:
            return username

    # None of the candidates matched KAGGLE_USERNAME; env is left on the last one tried.
    return last_username


def ensure_metadata_id(bundle_dir: Path, dataset_id: str):
    """Keep dataset-metadata.json's "id" in sync with the target dataset (documentation only;
    kagglehub.dataset_upload() takes the handle directly and doesn't read this file)."""
    meta_path = bundle_dir / "dataset-metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("id") != dataset_id:
        print(f"  Updating dataset-metadata.json id: {meta.get('id')!r} -> {dataset_id!r}")
        meta["id"] = dataset_id
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def upload_bundle(language: str, bundle_dir: Path, dataset_name: str, message: str, dry_run: bool, force: bool, current_user):
    dataset_id = f"{KAGGLE_USERNAME}/{dataset_name}"
    print(f"\n📤 {language.upper()}: {bundle_dir} -> {dataset_id}")

    if not bundle_dir.exists():
        print(f"❌ Bundle directory not found: {bundle_dir}")
        return False

    ensure_metadata_id(bundle_dir, dataset_id)

    user_ok = current_user == KAGGLE_USERNAME
    if current_user:
        print(f"  Authenticated as: {current_user}")
        if not user_ok:
            print(
                f"  ⚠️  Authenticated as '{current_user}', but dataset owner is "
                f"'{KAGGLE_USERNAME}'. This upload will fail unless '{current_user}' is a "
                f"collaborator on this dataset."
            )
    if not user_ok and not force and not dry_run:
        print(f"❌ Skipping {language} upload (auth mismatch). Re-run with --force to attempt anyway.")
        return False

    if dry_run:
        print(f"   (dry-run) would call: kagglehub.dataset_upload({dataset_id!r}, {str(bundle_dir)!r}, version_notes={message!r})")
        return True

    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed in this Python environment. Run with venv/bin/python3.")
        return False

    try:
        kagglehub.dataset_upload(dataset_id, str(bundle_dir), version_notes=message)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Upload failed for {language}: {e}")
        return False

    print(f"✅ {language} uploaded successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["telugu", "bhojpuri", "both"], default="both")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Version notes for kagglehub.dataset_upload")
    parser.add_argument("--dry-run", action="store_true", help="Print what would upload, no network calls")
    parser.add_argument("--force", action="store_true", help="Attempt upload even if the authenticated user doesn't match the dataset owner")
    args = parser.parse_args()

    current_user = resolve_kaggle_token()

    languages = ["telugu", "bhojpuri"] if args.lang == "both" else [args.lang]

    results = {}
    for lang in languages:
        info = BUNDLES[lang]
        results[lang] = upload_bundle(
            lang, info["bundle_dir"], info["dataset_name"], args.message, args.dry_run, args.force, current_user
        )

    print()
    failed = [lang for lang, ok in results.items() if not ok]
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
        sys.exit(1)
    print("✅ All uploads complete" if not args.dry_run else "✅ Dry run complete")


if __name__ == "__main__":
    main()
