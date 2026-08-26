"""
PackedLMDataset: reads from .bin token file, samples contiguous chunks for LM pretraining.
"""

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class PackedLMDataset(Dataset):
    """
    Reads pre-tokenized binary corpus and yields contiguous windows for LM training.
    Window size = max_seq_len + 1 (input + target), returned as (x[:-1], y[1:]).
    """

    def __init__(self, bin_path: str, max_seq_len: int = 512):
        """
        Args:
            bin_path: path to .bin file (uint32 token IDs)
            max_seq_len: sequence length (number of tokens in input x)
        """
        self.bin_path = Path(bin_path)
        self.max_seq_len = max_seq_len

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        self.tokens = np.memmap(self.bin_path, dtype=np.uint32, mode="r")
        self.num_tokens = len(self.tokens)

        if self.num_tokens < self.max_seq_len + 1:
            raise ValueError(
                f"Binary file too small ({self.num_tokens} tokens) "
                f"for seq_len {self.max_seq_len}+1"
            )

        self.num_samples = max(0, self.num_tokens - self.max_seq_len)
        print(
            f"PackedLMDataset: {self.num_tokens:,} tokens, "
            f"{self.num_samples:,} samples (seq_len={self.max_seq_len})"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Return contiguous chunk [idx, idx+max_seq_len+1).
        x = tokens[idx : idx+max_seq_len]
        y = tokens[idx+1 : idx+max_seq_len+1]  (shift-right target)
        """
        start = idx
        end = idx + self.max_seq_len + 1

        chunk = self.tokens[start:end]
        x = chunk[:-1].astype(np.int64)
        y = chunk[1:].astype(np.int64)

        return {"input_ids": x, "labels": y}
