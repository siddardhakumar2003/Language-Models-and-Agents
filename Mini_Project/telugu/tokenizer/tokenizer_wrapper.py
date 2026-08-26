"""
Telugu WordPiece Tokenizer Wrapper
Loads pre-trained WordPiece tokenizer and exposes encode/decode/vocab_size.
"""

import json
import os
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class TeluguTokenizer:
    def __init__(self, config_path=None, tokenizer_path=None):
        """
        Initialize tokenizer.

        Args:
            config_path: Path to tokenizer_config.json (default: configs/tokenizer_config.json)
            tokenizer_path: Path to wordpiece tokenizer .json (default: from config)
        """
        script_dir = Path(__file__).parent.parent

        if config_path is None:
            config_path = script_dir / "configs" / "tokenizer_config.json"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        if tokenizer_path is None:
            tokenizer_rel = self.config.get("tokenizer_path", "full_wordPiece_level/telugu_wp_tokenizer.json")
            tokenizer_path = script_dir / "tokenizer" / tokenizer_rel
        else:
            tokenizer_path = Path(tokenizer_path)

        self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.config["vocab_size"]
        self.bos_token_id = self.config["special_token_ids"]["bos_token_id"]
        self.eos_token_id = self.config["special_token_ids"]["eos_token_id"]
        self.pad_token_id = self.config["special_token_ids"]["pad_token_id"]
        self.unk_token_id = self.config["special_token_ids"]["unk_token_id"]

    def encode(self, text: str, add_special_tokens=False):
        """Encode text to token IDs."""
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def encode_batch(self, texts, add_special_tokens=False):
        """Encode multiple texts (vectorized)."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self):
        """Return vocabulary size."""
        return self.vocab_size


if __name__ == "__main__":
    tok = TeluguTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"BOS ID: {tok.bos_token_id}, EOS ID: {tok.eos_token_id}, PAD ID: {tok.pad_token_id}")

    text = "ఇది తెలుగు భాష"
    ids = tok.encode(text)
    print(f"Text: {text}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {tok.decode(ids)}")
