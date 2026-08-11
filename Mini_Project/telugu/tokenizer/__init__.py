"""
Telugu BPE Tokenizer

Vocabulary size: 32,000 tokens
Training data: telugu.txt (cleaned text)
Output: telugu_tokenizer.json
"""

from pathlib import Path
from tokenizers import Tokenizer

def load_tokenizer():
    """Load trained Telugu tokenizer"""
    tokenizer_path = Path(__file__).parent / "telugu_tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    return Tokenizer.from_file(str(tokenizer_path))

__all__ = ['load_tokenizer']
