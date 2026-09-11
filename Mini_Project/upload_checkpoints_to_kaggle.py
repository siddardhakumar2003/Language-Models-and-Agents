#!/usr/bin/env python3
"""
Upload trained checkpoints to Kaggle datasets, using the `kagglehub` Python client:

  telugu   -> kspsvln/checkpoint-telugu
  bhojpuri -> kspsvln/checkpoint-bhojpuri

Local source for each language is <language>/model/outputs/ (matching the
dataset-metadata.json already committed there, whose "resources" list
checkpoints/checkpoint_best.pt and checkpoints/checkpoint_last.pt) -- so the
uploaded dataset root ends up containing a checkpoints/ folder directly,
matching pretrain_kaggle.ipynb's own resume convention:
    CHECK_DIR = "/kaggle/input/checkpoint-<language>/checkpoints"

Requires the `kagglehub` package (only present in this repo's venv/):
    venv/bin/python3 upload_checkpoints_to_kaggle.py telugu

Usage:
    python3 upload_checkpoints_to_kaggle.py both                  # both languages
    python3 upload_checkpoints_to_kaggle.py telugu
    python3 upload_checkpoints_to_kaggle.py bhojpuri --message "Custom note"
    python3 upload_checkpoints_to_kaggle.py telugu --dry-run       # show what would upload, no network calls
    python3 upload_checkpoints_to_kaggle.py telugu --force         # skip the auth-owner check

Auth: same as upload_phase2_bundles.py -- tries every KAGGLE_API_TOKEN found in
api_token.txt (there's more than one, for different accounts) and uses whichever
one actually authenticates as KAGGLE_USERNAME.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

KAGGLE_USERNAME = "kspsvln"

CHECKPOINTS = {
    "telugu": {
        "local_dir": REPO_ROOT / "telugu" / "model" / "outputs",
        "dataset_name": "checkpoint-telugu",
    },
    "bhojpuri": {
        "local_dir": REPO_ROOT / "bhojpuri" / "model" / "outputs",
        "dataset_name": "checkpoint-bhojpuri",
    },
}

DEFAULT_MESSAGE = "Updated checkpoint"


def find_kaggle_tokens():
    """Return every KAGGLE_API_TOKEN value found in api_token.txt, in file order."""
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

    return last_username


def ensure_metadata_id(local_dir: Path, dataset_id: str):
    """Keep dataset-metadata.json's "id" in sync with the target dataset (documentation only;
    kagglehub.dataset_upload() takes the handle directly and doesn't read this file)."""
    meta_path = local_dir / "dataset-metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("id") != dataset_id:
        print(f"  Updating dataset-metadata.json id: {meta.get('id')!r} -> {dataset_id!r}")
        meta["id"] = dataset_id
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def describe_checkpoints(local_dir: Path):
    """Print what's actually sitting in <local_dir>/checkpoints/ so a stale checkpoint
    doesn't get uploaded silently -- eyeball the timestamps/sizes before trusting this."""
    ckpt_dir = local_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"  ❌ No checkpoints/ folder at {ckpt_dir}")
        return False
    found = False
    for name in ["checkpoint_best.pt", "checkpoint_last.pt"]:
        p = ckpt_dir / name
        if p.exists():
            found = True
            size_mb = p.stat().st_size / 1e6
            from datetime import datetime
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            print(f"  {name}: {size_mb:.1f} MB, modified {mtime}")
        else:
            print(f"  {name}: missing")
    return found


def upload_checkpoint(language: str, local_dir: Path, dataset_name: str, message: str, dry_run: bool, force: bool, current_user):
    dataset_id = f"{KAGGLE_USERNAME}/{dataset_name}"
    print(f"\n📤 {language.upper()}: {local_dir} -> {dataset_id}")

    if not local_dir.exists():
        print(f"❌ {local_dir} does not exist yet -- nothing to upload.")
        return False

    has_checkpoints = describe_checkpoints(local_dir)
    if not has_checkpoints:
        print(f"❌ No checkpoint_best.pt / checkpoint_last.pt found under {local_dir}/checkpoints/ -- nothing to upload.")
        return False

    ensure_metadata_id(local_dir, dataset_id)

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
        print(f"   (dry-run) would call: kagglehub.dataset_upload({dataset_id!r}, {str(local_dir)!r}, version_notes={message!r})")
        return True

    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed in this Python environment. Run with venv/bin/python3.")
        return False

    try:
        kagglehub.dataset_upload(dataset_id, str(local_dir), version_notes=message)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Upload failed for {language}: {e}")
        return False

    print(f"✅ {language} checkpoint uploaded successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lang", choices=["telugu", "bhojpuri", "both"], help="Which language's checkpoint to upload")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Version notes for kagglehub.dataset_upload")
    parser.add_argument("--dry-run", action="store_true", help="Print what would upload, no network calls")
    parser.add_argument("--force", action="store_true", help="Attempt upload even if the authenticated user doesn't match the dataset owner")
    args = parser.parse_args()

    current_user = resolve_kaggle_token()

    languages = ["telugu", "bhojpuri"] if args.lang == "both" else [args.lang]

    results = {}
    for lang in languages:
        info = CHECKPOINTS[lang]
        results[lang] = upload_checkpoint(
            lang, info["local_dir"], info["dataset_name"], args.message, args.dry_run, args.force, current_user
        )

    print()
    failed = [lang for lang, ok in results.items() if not ok]
    if failed:
        print(f"❌ Not uploaded: {', '.join(failed)}")
        sys.exit(1)
    print("✅ All uploads complete" if not args.dry_run else "✅ Dry run complete")


if __name__ == "__main__":
    main()
