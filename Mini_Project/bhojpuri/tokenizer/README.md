# Bhojpuri BPE Tokenizer

## Overview

**Type**: Byte Pair Encoding (BPE)  
**Vocabulary Size**: 16,000 tokens  
**Language**: Bhojpuri (Lower-resource)  
**Training Data**: `bhoj.txt` (1,580 cleaned texts, 1.9 MB)

---

## Quick Start

### Train Tokenizer

```bash
cd /home/ubuntu/LMA/Mini_Project/bhojpuri

python3 tokenizer/train_tokenizer.py
```

**Expected Output:**
- `tokenizer/bhoj_tokenizer.json` (tokenizer file)
- `tokenizer/tokenizer_config.json` (configuration)

**Duration**: 2-5 minutes (depends on data size and CPU)

---

## Using the Tokenizer

### Load Tokenizer

```python
from tokenizers import Tokenizer

# Load from file
tokenizer = Tokenizer.from_file("tokenizer/bhoj_tokenizer.json")

# Or use the helper function
from bhojpuri.tokenizer import load_tokenizer
tokenizer = load_tokenizer()
```

### Encode Text

```python
# Single text
text = "भारत एक महान देश है।"
encoded = tokenizer.encode(text)

print(f"Tokens: {encoded.ids}")
print(f"Token count: {len(encoded.ids)}")
print(f"Token strings: {encoded.tokens}")
```

### Decode Tokens

```python
# Decode token IDs back to text
token_ids = [100, 200, 300, ...]
decoded = tokenizer.decode(token_ids)
print(f"Decoded text: {decoded}")
```

### Batch Processing

```python
# Encode multiple texts
texts = [
    "भारत दक्षिण एशिया में स्थित है।",
    "हिन्दी एक मुख्य भाषा है।",
    "यह एक परीक्षण है।"
]

encoded_batch = tokenizer.encode_batch(texts)

for text, encoded in zip(texts, encoded_batch):
    print(f"{text}")
    print(f"  Tokens: {len(encoded.ids)}")
```

---

## Tokenizer Specifications

### Vocabulary

- **Size**: 16,000 tokens
- **Type**: BPE (Byte Pair Encoding)
- **Model**: ByteLevel BPE

### Special Tokens

| Token | ID | Purpose |
|-------|----|---------| 
| `<pad>` | 0 | Padding (for batching) |
| `<unk>` | 1 | Unknown token |
| `<cls>` | 2 | Sequence start |
| `<sep>` | 3 | Sequence separator |
| `<mask>` | 4 | Masked token |
| `<bos>` | 5 | Beginning of sequence |
| `<eos>` | 6 | End of sequence |

---

## Configuration

File: `tokenizer_config.json`

```json
{
  "language": "Bhojpuri",
  "vocab_size": 16000,
  "tokenizer_type": "BPE",
  "model": "ByteLevel BPE",
  "special_tokens": ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<bos>", "<eos>"],
  "training_file": "data/bhoj.txt",
  "training_file_size_mb": 1.9,
  "created_at": "tokenizer"
}
```

---

## Training Details

### Input Data

- **File**: `data/bhoj.txt`
- **Format**: One text per line (UTF-8)
- **Size**: 1.9 MB
- **Lines**: 1,580 cleaned texts

### Training Process

1. **Normalization**: Unicode NFD + lowercase
2. **Pre-tokenization**: Byte-level splitting
3. **BPE Training**: Learn 16K merge rules
4. **Serialization**: Save to JSON format

### Parameters

```python
vocab_size = 16000
min_frequency = 2
show_progress = True
special_tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<bos>", "<eos>"]
```

---

## Example Usage in Model Training

```python
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(text)
        
        # Pad or truncate
        ids = encoded.ids[:self.max_length]
        ids += [self.tokenizer.token_to_id("<pad>")] * (self.max_length - len(ids))
        
        return {
            'input_ids': ids,
            'attention_mask': [1 if x != 0 else 0 for x in ids]
        }

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer/bhoj_tokenizer.json")

# Create dataset
dataset = TextDataset("data/bhoj.txt", tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in model training
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    # ... model forward pass ...
```

---

## Performance Notes

### Training Time
- **Duration**: 2-5 minutes (CPU dependent)
- **Data Size**: 1.9 MB input
- **Vocabulary**: 16K tokens

### Tokenization Speed
- **Speed**: ~10,000-50,000 tokens/second (CPU dependent)
- **Memory**: Minimal (~50-100 MB)

### Token Count Stats

For typical Bhojpuri text:
- **Short text** (< 50 chars): 5-10 tokens
- **Medium text** (50-200 chars): 15-40 tokens
- **Long text** (> 200 chars): 40+ tokens

---

## Troubleshooting

### "Tokenizer file not found"

```bash
# Make sure you trained the tokenizer first
python3 tokenizer/train_tokenizer.py
```

### "Training data not found"

```bash
# Make sure you cleaned the data first
python3 data_collect/clean_data.py
```

### "ModuleNotFoundError: No module named 'tokenizers'"

```bash
pip install tokenizers --break-system-packages
```

### Special token not recognized

```python
# Get token ID for special token
pad_id = tokenizer.token_to_id("<pad>")
unk_id = tokenizer.token_to_id("<unk>")
```

---

## Next Steps

1. ✅ **Train tokenizer**: `python3 train_tokenizer.py`
2. ✅ **Verify output**: Check `bhoj_tokenizer.json` exists
3. → **Train model**: `cd ../train && python3 train.py`
4. → **Evaluate**: `cd ../eval && python3 evaluate.py`

---

## Reference

- **Tokenizers Library**: https://github.com/huggingface/tokenizers
- **BPE Algorithm**: Sennrich et al., 2016
- **Byte-Level BPE**: Wang & Komatsuzaki, 2019

---

**Status**: Ready to use ✅
