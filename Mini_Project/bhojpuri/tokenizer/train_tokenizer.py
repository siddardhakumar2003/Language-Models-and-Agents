#!/usr/bin/env python3
"""
Train BPE tokenizer for Bhojpuri language

Vocabulary size: 16K tokens (chosen for lower-resource language)
Input: bhojpuri/data/bhoj.txt (cleaned training data)
Output: bhojpuri/tokenizer/bhoj_tokenizer.json
"""

import json
import logging
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_bpe_tokenizer():
    """Create BPE tokenizer for Bhojpuri"""

    logger.info("Creating BPE tokenizer for Bhojpuri...")

    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())

    # Set normalizer (Unicode normalization)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase()
    ])

    # Set pre-tokenizer (character-level)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Set decoder (byte-level decoding)
    tokenizer.decoder = decoders.ByteLevel()

    # Set post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    return tokenizer

def train_tokenizer():
    """Train BPE tokenizer on Bhojpuri data"""

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Input and output paths
    training_file = project_root / "data" / "bhoj.txt"
    output_file = script_dir / "bhoj_tokenizer.json"
    config_file = script_dir / "tokenizer_config.json"

    print("=" * 80)
    print("BHOJPURI BPE TOKENIZER TRAINING")
    print("=" * 80)
    print(f"Training file: {training_file}")
    print(f"Output file: {output_file}")
    print(f"Vocabulary size: 16,000 tokens")
    print("=" * 80)

    # Check if training file exists
    if not training_file.exists():
        logger.error(f"Training file not found: {training_file}")
        logger.error("Make sure to run cleaning first: python3 data_collect/clean_data.py")
        return False

    # Check file size
    file_size_mb = training_file.stat().st_size / (1024 * 1024)
    logger.info(f"Training file size: {file_size_mb:.2f} MB")

    # Create tokenizer
    tokenizer = create_bpe_tokenizer()

    # Train tokenizer
    logger.info("\nTraining tokenizer...")
    trainer = trainers.BpeTrainer(
        vocab_size=16000,  # 16K vocabulary for lower-resource language
        min_frequency=2,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<cls>",
            "<sep>",
            "<mask>",
            "<bos>",
            "<eos>"
        ],
        show_progress=True
    )

    # Train on file
    tokenizer.train([str(training_file)], trainer=trainer)

    # Save tokenizer
    logger.info(f"\nSaving tokenizer to {output_file}...")
    tokenizer.save(str(output_file))
    logger.info("✓ Tokenizer saved successfully")

    # Save configuration
    config = {
        "language": "Bhojpuri",
        "vocab_size": 16000,
        "tokenizer_type": "BPE",
        "model": "ByteLevel BPE",
        "special_tokens": [
            "<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<bos>", "<eos>"
        ],
        "training_file": str(training_file),
        "training_file_size_mb": file_size_mb,
        "created_at": str(Path(output_file).parent)
    }

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Configuration saved to {config_file}")

    # Test tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("TOKENIZER VERIFICATION")
    logger.info("=" * 80)

    test_text = "भारत एक महान देश है।"
    tokens = tokenizer.encode(test_text)

    logger.info(f"\nTest text: {test_text}")
    logger.info(f"Encoded tokens: {tokens.ids}")
    logger.info(f"Number of tokens: {len(tokens.ids)}")
    logger.info(f"Token string: {tokens.tokens}")

    # Decode to verify
    decoded = tokenizer.decode(tokens.ids)
    logger.info(f"Decoded text: {decoded}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ TOKENIZER TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nTokenizer saved: {output_file}")
    logger.info(f"Vocabulary size: 16,000 tokens")
    logger.info(f"Ready for model training!")
    logger.info("=" * 80 + "\n")

    return True

def load_and_test_tokenizer():
    """Load and test the trained tokenizer"""

    script_dir = Path(__file__).resolve().parent
    tokenizer_file = script_dir / "bhoj_tokenizer.json"

    if not tokenizer_file.exists():
        logger.error(f"Tokenizer file not found: {tokenizer_file}")
        return False

    logger.info(f"\nLoading tokenizer from {tokenizer_file}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # Test encoding/decoding
    test_texts = [
        "भारत दक्षिण एशिया में स्थित है।",
        "हिन्दी एक मुख्य भाषा है।",
        "यह एक परीक्षण है।"
    ]

    logger.info("\nTokenizer Test Results:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        logger.info(f"  Text: {text}")
        logger.info(f"  Tokens: {len(encoded.ids)} tokens")
        logger.info(f"  Decoded: {decoded}")
        logger.info()

    return True

if __name__ == "__main__":
    import sys

    # Train tokenizer
    success = train_tokenizer()

    if success:
        # Test loaded tokenizer
        load_and_test_tokenizer()
        sys.exit(0)
    else:
        logger.error("Tokenizer training failed")
        sys.exit(1)
