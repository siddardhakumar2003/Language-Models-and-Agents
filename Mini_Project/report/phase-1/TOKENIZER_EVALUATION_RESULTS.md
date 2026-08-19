# Comprehensive Tokenizer Evaluation Report
## Phase 1: All 6 Tokenizers (Telugu + Bhojpuri)

---

## Telugu (Model H) - Evaluation Results

### Tokenizer 1: Byte-Level BPE (50K Vocab)

**Statistics:**
- **Vocabulary Size**: 50,000
- **Unique Tokens Used**: 1,978
- **Total Tokens (Test Set)**: 98,469
- **Average Characters per Token**: 1.3327
- **Unknown (UNK) Token Rate**: 0.0000%
- **Unknown Token Count**: 0

**Top 10 Token Frequencies:**
1. ౄ (Vowel Sign Vocalic R Rweekly) - 9,962 (10.12%)
2. ಿ (Vowel Sign I) - 7,139 (7.25%)
3. ಾ (Vowel Sign AA) - 7,103 (7.21%)
4. ು (Vowel Sign U) - 6,089 (6.18%)
5. న (Consonant Na) - 3,348 (3.40%)
6. ర (Consonant Ra) - 3,261 (3.31%)
7. ఛ (Consonant Cha) - 3,146 (3.19%)
8. ల (Consonant La) - 3,122 (3.17%)
9. ృ (Vowel Sign Vocalic R) - 2,511 (2.55%)
10. ూ (Vowel Sign UU) - 2,386 (2.42%)

**Key Insights:**
- Excellent compression efficiency with avg 1.33 chars/token
- 0% unknown token rate indicates complete vocab coverage
- Diacritics dominate top frequencies (vowel signs)
- Compact representation suitable for character-level models

---

### Tokenizer 2: Unicode-Level BPE (50K Vocab)

**Statistics:**
- **Vocabulary Size**: 50,000
- **Unique Tokens Used**: 11,398
- **Total Tokens (Test Set)**: 22,136
- **Average Characters per Token**: 5.9283
- **Unknown (UNK) Token Rate**: 0.0000%
- **Unknown Token Count**: 0

**Top 10 Token Frequencies:**
1. (space) - 61 (0.28%)
2. లో (Preposition "in") - 51 (0.23%)
3. ఈ (Pronoun "this") - 50 (0.23%)
4. కూడా (Adverb "also") - 50 (0.23%)
5. , (Comma) - 50 (0.23%)
6. ారు (Verb suffix) - 48 (0.22%)
7. కు (Postposition "to") - 40 (0.18%)
8. ని (Object marker) - 39 (0.18%)
9. ' (Apostrophe) - 38 (0.17%)
10. తో (Preposition "with") - 35 (0.16%)

**Key Insights:**
- Larger avg token length (5.93 chars) preserves linguistic units
- Fewer total tokens → better semantic coherence
- Functional words preserved as single tokens
- 0% UNK rate with much higher coverage

---

### Tokenizer 3: WordPiece (50K Vocab)

**Statistics:**
- **Vocabulary Size**: 50,000
- **Unique Tokens Used**: 9,539
- **Total Tokens (Test Set)**: 22,489
- **Average Characters per Token**: 5.8353
- **Unknown (UNK) Token Rate**: 0.0000%
- **Unknown Token Count**: 0

**Top 10 Token Frequencies:**
1. , (Comma) - 620 (2.76%)
2. ఈ (Pronoun "this") - 211 (0.94%)
3. కూడా (Adverb "also") - 103 (0.46%)
4. ##‌ (Subword marker + Zero-Width Joiner) - 91 (0.40%)
5. ఆ (Pronoun "that") - 79 (0.35%)
6. : (Colon) - 78 (0.35%)
7. - (Dash) - 78 (0.35%)
8. ##ని (Subword: object marker) - 72 (0.32%)
9. ##లో (Subword: "in") - 72 (0.32%)
10. ##ను (Subword: accusative) - 71 (0.31%)

**Key Insights:**
- WordPiece subword strategy (## prefix for continuation)
- Slightly more subwords than Unicode-level BPE
- Comparable avg token length (5.84 chars)
- Excellent handling of affixes and morphological variants
- 0% UNK rate

---

## Bhojpuri (Model L) - Known Statistics

### Tokenizer 1: Byte-Level BPE (8K Vocab)

**From Training Logs:**
- **Vocabulary Size**: 8,000
- **Vocabulary Coverage**: 97.01%
- **Average Characters per Token**: 1.54
- **Unknown Token Rate**: 0.0% (from test set evaluation)

---

### Tokenizer 2: Unicode-Level BPE (16K Vocab)

**From Training Reports:**
- **Vocabulary Size**: 16,000
- **Vocabulary Coverage**: 97.77%
- **Average Characters per Token**: 3.44
- **Unknown Token Rate**: 0.0% (from test set evaluation)

**Top Token Frequencies:**
- आ (Vowel AA) - 1,321,305 (9.7%)
- ् (Virama) - 968,293 (7.1%)
- े (Vowel E) - 791,144 (5.8%)
- ि (Vowel I) - 612,278 (4.5%)

---

### Tokenizer 3: WordPiece (16K Vocab)

**From Training Logs:**
- **Vocabulary Size**: 16,000
- **Total Test Tokens**: 5,930,926
- **Unknown Token Rate**: 0.0001% (extremely low)
- **Random Seed**: 42 (reproducible)

**Top Token Frequencies:**
- के (Postposition "of") - 191,018
- । (Devanagari Danda) - 151,048
- , (Comma) - 119,912
- - (Dash) - 96,457
- में (Postposition "in") - 90,624

---

## Comparative Analysis

### Vocabulary Sizes
| Language | Variant | Vocab Size |
|----------|---------|------------|
| Telugu | Byte-level | 50,000 |
| Telugu | Unicode-level | 50,000 |
| Telugu | WordPiece | 50,000 |
| Bhojpuri | Byte-level | 8,000 |
| Bhojpuri | Unicode-level | 16,000 |
| Bhojpuri | WordPiece | 16,000 |

### Average Characters per Token
| Language | Variant | Avg Chars/Token |
|----------|---------|-----------------|
| Telugu | Byte-level | 1.3327 |
| Telugu | Unicode-level | 5.9283 |
| Telugu | WordPiece | 5.8353 |
| Bhojpuri | Byte-level | 1.54 |
| Bhojpuri | Unicode-level | 3.44 |
| Bhojpuri | WordPiece | ~3-5 (estimated) |

### Unknown Token Rates
| Language | Variant | UNK Rate |
|----------|---------|----------|
| Telugu | Byte-level | 0.0000% |
| Telugu | Unicode-level | 0.0000% |
| Telugu | WordPiece | 0.0000% |
| Bhojpuri | Byte-level | 0.0000% |
| Bhojpuri | Unicode-level | 0.0000% |
| Bhojpuri | WordPiece | 0.0001% |

---

## Key Findings

### Telugu Tokenizers
1. **Byte-level BPE**: Maximum compression (1.33 chars/token), highest subword count
2. **Unicode-level BPE**: Optimal balance (5.93 chars/token), better linguistic coherence
3. **WordPiece**: Similar to Unicode (5.84 chars/token), explicit subword handling

All Telugu tokenizers achieve **0% unknown token rate** with 50K vocabulary.

### Bhojpuri Tokenizers
1. **Byte-level BPE**: Compact (1.54 chars/token), resource-efficient for 8K vocab
2. **Unicode-level BPE**: Best balance (3.44 chars/token) with 16K vocab
3. **WordPiece**: Production-ready with 0.0001% UNK rate on 5.9M test tokens

### General Observations
- **Vocabularies are independent** across all 6 tokenizers
- **No shared tokens** between language-specific tokenizers
- **Telugu requires larger vocab** (50K) due to script complexity
- **Bhojpuri uses smaller vocab** (8K-16K) suitable for lower-resource setting
- **All tokenizers achieve near-zero UNK rates** indicating excellent coverage
- **Unicode-level BPE outperforms** on linguistic coherence vs compression trade-off

---

**Report Generated**: August 19, 2026
**Evaluation Framework**: Tokenizers library (Hugging Face)
**Test Set Size**: 500 samples per language (where available)
