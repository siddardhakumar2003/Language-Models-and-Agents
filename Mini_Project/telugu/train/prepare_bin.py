"""
Prepare binary token corpus from Telugu training data.
Reads all *.txt files in data/{split}/, tokenizes, appends EOS, writes to .bin + metadata.
Supports resumable runs (tracks progress).
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.tokenizer_wrapper import TeluguTokenizer


def prepare_bin(split: str = "train", output_dir: str = None, sample_lines: int = None, batch_size: int = 1000):
    """
    Tokenize all .txt files in data/{split}/ and write to .bin + metadata.

    Args:
        split: 'train', 'val', or 'test'
        output_dir: where to save .bin and metadata (default: data/)
        sample_lines: if set, only process first N lines (for testing)
        batch_size: tokenize this many lines at once for efficiency
    """
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / split

    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split}.bin"
    metadata_path = output_dir / f"{split}_metadata.json"

    tokenizer = TeluguTokenizer()
    eos_id = tokenizer.eos_token_id
    print(f"Loading tokenizer: vocab_size={tokenizer.vocab_size}, EOS={eos_id}")

    txt_files = sorted(data_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in {data_dir}")
    for f in txt_files:
        print(f"  - {f.name}")

    metadata = {"split": split, "num_tokens": 0, "processed_lines": 0}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Resuming from checkpoint: {metadata['processed_lines']} lines already processed")

    all_token_ids = []
    processed_count = 0

    file_line_counts = {}
    cumulative_lines = 0
    for f in txt_files:
        num_lines = sum(1 for _ in open(f, "r", encoding="utf-8", errors="ignore"))
        file_line_counts[f.name] = num_lines
        cumulative_lines += num_lines

    print(f"Total lines across all files: {cumulative_lines}")

    for txt_file in txt_files:
        print(f"\nProcessing {txt_file.name}...")

        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            batch = []
            pbar = tqdm(f, total=file_line_counts[txt_file.name], desc=txt_file.name)

            for line in pbar:
                processed_count += 1

                if processed_count <= metadata.get("processed_lines", 0):
                    continue

                line = line.strip()
                if not line:
                    continue

                batch.append(line)

                if len(batch) >= batch_size:
                    ids_batch = tokenizer.encode_batch(batch, add_special_tokens=False)
                    for ids in ids_batch:
                        all_token_ids.extend(ids)
                        all_token_ids.append(eos_id)
                    batch = []

                if sample_lines and metadata["processed_lines"] >= sample_lines:
                    break

            if batch:
                ids_batch = tokenizer.encode_batch(batch, add_special_tokens=False)
                for ids in ids_batch:
                    all_token_ids.extend(ids)
                    all_token_ids.append(eos_id)

        pbar.close()

        metadata["processed_lines"] = processed_count
        metadata["num_tokens"] = len(all_token_ids)

        print(f"  Processed lines so far: {processed_count}, total tokens: {len(all_token_ids):,}")

        if sample_lines and metadata["processed_lines"] >= sample_lines:
            print(f"Reached sample limit of {sample_lines} lines")
            break

    token_array = np.array(all_token_ids, dtype=np.uint32)
    token_array.tofile(bin_path)

    metadata["num_tokens"] = len(all_token_ids)
    metadata["vocab_size"] = tokenizer.vocab_size
    metadata["tokenizer_class"] = "TeluguTokenizer"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Tokenized {metadata['processed_lines']:,} lines into {metadata['num_tokens']:,} tokens")
    print(f"  Written to: {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None, help="output directory (default: data/)")
    parser.add_argument("--sample-lines", type=int, default=None, help="only process N lines (for testing)")
    parser.add_argument("--batch-size", type=int, default=1000, help="batch size for tokenization")
    args = parser.parse_args()

    prepare_bin(args.split, args.output_dir, args.sample_lines, args.batch_size)
