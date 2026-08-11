"""
Bhojpuri BPE Tokenizer

Vocabulary size: 16,000 tokens
Training data: bhoj.txt (cleaned text)
Output: bhoj_tokenizer.json
"""

from pathlib import Path
from tokenizers import Tokenizer

def load_tokenizer():
    """Load trained Bhojpuri tokenizer"""
    tokenizer_path = Path(__file__).parent / "bhoj_tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    return Tokenizer.from_file(str(tokenizer_path))

__all__ = ['load_tokenizer']
