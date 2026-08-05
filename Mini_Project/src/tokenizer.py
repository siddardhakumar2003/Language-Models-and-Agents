from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import unicodedata
import re
import hashlib


# Load Telugu IndicCorpV2 in streaming mode
dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    "indiccorp_v2",
    data_files={
        "train": "data/te.txt"
    },
    streaming=True
)


# Text cleaning
def clean_text(text):
    # Unicode normalization (important for Telugu)
    text = unicodedata.normalize("NFC", text)

    # Remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    return text.strip()


# Iterator for tokenizer training
seen = set()

def batch_iterator():
    for sample in dataset["train"]:
        text = clean_text(sample["text"])

        # Skip empty text
        if not text:
            continue

        # Remove exact duplicates
        h = hashlib.md5(text.encode("utf-8")).hexdigest()

        if h in seen:
            continue

        seen.add(h)

        yield text


# Create BPE tokenizer
tokenizer = Tokenizer(models.BPE())


# Telugu note:
# ByteLevel works, but for Telugu-only models
# you may want Whitespace or Metaspace later.
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()


# BPE trainer
trainer = trainers.BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    ]
)


# Train tokenizer
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer
)


# Save tokenizer
tokenizer.save("telugu_tokenizer.json")


print("Tokenizer training completed")
print("Vocabulary size:", tokenizer.get_vocab_size())