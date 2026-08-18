# Bhojpuri Corpus Sources & Collection Methodology

**Last Updated**: 2026-08-15  
**Project**: LMA Individual Mini-Project (Monolingual Transformer LM in Bhojpuri)

---

## Summary

This document records all data sources added to the Bhojpuri corpus during Phase 2 expansion
(August 2026), including their collection method (automated vs. manual), original format,
size, license, and how each is used in the pretraining pipeline.

The goal was to expand beyond the initial Phase 1 corpus (776,801 lines / ~135.6M tokens,
mostly Hindi proxy + web scraping + HF corpus) by incorporating **authentic Bhojpuri texts**
from multiple independent sources and adding **genuine Tesseract OCR** of scanned books (not
just pre-extracted text).

---

## Data Sources

### Pre-Built Corpora (Automated Ingestion)

These are existing, assembled datasets that are downloaded, converted to our JSONL format,
cleaned, deduplicated, and merged into the splits.

#### 1. bho.wikipedia.org Dump (Authentic Bhojpuri Wikipedia)

| Property | Value |
|----------|-------|
| **Source Name** | Bhojpuri Wikipedia (bho.wikipedia.org) |
| **URL** | https://dumps.wikimedia.org/bhwiki/latest/bhwiki-latest-pages-articles.xml.bz2 |
| **License** | CC BY-SA 3.0 |
| **Collection Method** | **Automated**: `download_wikipedia_dump.py` |
| **Format** | XML dump (compressed with bz2) |
| **Original Size** | ~19.7 MB compressed (~8,857 articles) |
| **Coverage** | Native Bhojpuri Wikipedia articles (written by Bhojpuri Wikipedia community) |
| **Pipeline** | Download → parse XML → strip wikitext markup → `{"text": ...}` JSONL → clean → dedup → merge |
| **Raw JSONL Stage** | `bhojpuri/data/wiki_raw/wiki_batch_*.jsonl` |
| **Cleaned JSONL Stage** | `bhojpuri/data/wiki_cleaned/cleaned_*.jsonl` (archived after merge) |
| **Rows After Cleaning** | [see config.json `token_progress` for current total] |
| **Deduplication** | Against root `bhoj.txt` accumulator (exact MD5 + near-match) |
| **Import Note** | This is **genuine** Bhojpuri Wikipedia, not a Hindi proxy. The prior Phase 1 corpus used Hindi Wikipedia as a script-proxy; this dump is authenticated Bhojpuri. |

#### 2. HuggingFace FineWeb-2 bho_Deva Subset

| Property | Value |
|----------|-------|
| **Source Name** | HuggingFace `fineweb-2` (bho_Deva language config) |
| **URL** | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **License** | CC0 (public domain) |
| **Collection Method** | **Automated**: `download_fineweb2_corpus.py` |
| **Format** | Parquet (downloaded via direct HTTPS, not `datasets` library) |
| **Original Size** | ~18.7k documents (CommonCrawl-derived web text) |
| **Coverage** | Bhojpuri web text scraped from Common Crawl snapshots (2018–2025) |
| **Pipeline** | Download parquet → `pd.read_parquet()` → `{"text": ...}` JSONL → clean → dedup → merge |
| **Raw JSONL Stage** | `bhojpuri/data/fineweb2_raw/fineweb2_batch_*.jsonl` |
| **Cleaned JSONL Stage** | `bhojpuri/data/fineweb2_cleaned/cleaned_*.jsonl` (archived after merge) |
| **Rows After Cleaning** | [see config.json] |
| **Deduplication** | Against root accumulator (exact MD5 + near-match) |

#### 3. BHLTR Corpus (shashwatup9k/bho-resources)

| Property | Value |
|----------|-------|
| **Source Name** | BHLTR (Bhojpuri Language Resources) |
| **URL** | https://github.com/shashwatup9k/bho-resources |
| **License** | CC BY-NC-SA 4.0 (non-commercial) |
| **Collection Method** | **Automated**: `download_bhltr_corpus.py` (git clone + extract .bho files) |
| **Format** | Plain text `.bho` files (line = sentence/document) |
| **Original Size** | Academic monolingual + POS-annotated + treebank Bhojpuri texts |
| **Coverage** | High-quality manually annotated Bhojpuri (POS tags, syntactic structure) |
| **Pipeline** | Clone repo → extract `.bho` files from `mono-bho-corpus/` → `{"text": ...}` JSONL → clean → dedup → merge |
| **Raw JSONL Stage** | `bhojpuri/data/bhltr_raw/bhltr_batch_*.jsonl` |
| **Cleaned JSONL Stage** | `bhojpuri/data/bhltr_cleaned/cleaned_*.jsonl` (archived after merge) |
| **Rows After Cleaning** | [see config.json] |
| **Deduplication** | Against root accumulator |
| **License Note** | Non-commercial license (CC BY-NC-SA 4.0) is acceptable for this academic course project. If derivative works are published commercially, additional licensing review would be needed. |

#### 4. fossdot/bhojpuri-corpus (GitHub Community Corpus)

| Property | Value |
|----------|-------|
| **Source Name** | fossdot/bhojpuri-corpus |
| **URL** | https://github.com/fossdot/bhojpuri-corpus |
| **License** | CC BY-SA 4.0 (Data) |
| **Collection Method** | **Automated**: `download_github_corpus.py` (git clone + extract reviewed JSONL) |
| **Format** | JSONL (reviewed/completed status, speaker IDs) |
| **Original Size** | ~171 reviewed sentences (~62 KB) |
| **Coverage** | Community-contributed Bhojpuri text from native speakers |
| **Pipeline** | Clone repo → filter `data/reviewed/` JSONL to `status="completed"` entries → extract `text` field → `{"text": ...}` JSONL → clean → dedup → merge |
| **Raw JSONL Stage** | `bhojpuri/data/fossdot_raw/fossdot_batch_*.jsonl` |
| **Cleaned JSONL Stage** | `bhojpuri/data/fossdot_cleaned/cleaned_*.jsonl` (archived after merge) |
| **Rows After Cleaning** | [see config.json; likely negligible contribution (~50–100 lines post-cleaning)] |
| **Deduplication** | Against root accumulator |
| **Import Note** | Small dataset, but included as explicitly requested by the user and as a model for community-sourced data. |

---

### Self-Collected via Scraping & OCR

These sources are generated by running our own scraping/extraction scripts on third-party data.

#### 5. Archive.org Bhojpuri Books (Pre-Extracted djvu.txt)

| Property | Value |
|----------|-------|
| **Source Name** | archive.org (Bhojpuri-tagged books, pre-OCR'd content) |
| **Source URL** | https://archive.org/search.php (search query: `language:(Bhojpuri) AND mediatype:(texts)`) |
| **License** | Variable (items retain original book licenses; most are public domain or CC-licensed) |
| **Collection Method** | **Automated Script**: `download_archive_org_bhojpuri.py` |
| **Search Strategies** | 5 Bhojpuri-specific queries (vs. the original Hindi-proxy searches): |
| | - `language:(Bhojpuri) AND mediatype:(texts)` |
| | - `subject:(Bhojpuri) AND mediatype:(texts)` |
| | - `title:(Bhojpuri OR bhojpuri) AND mediatype:(texts)` |
| | - `subject:(Bhojpuri literature OR Bhojpuri language) AND mediatype:(texts)` |
| | - `description:(Bhojpuri) AND mediatype:(texts)` |
| **Format** | Pre-extracted text (items' internal `_djvu.txt` files, accessed via archive.org API) |
| **Extraction Method** | Via archive.org Scrape API + metadata API (no true OCR here; reusing archive.org's existing OCR) |
| **Target Items** | Up to 500 archive.org items (configurable via `--target-items` flag) |
| **Pipeline** | Search archive.org → download pre-extracted `_djvu.txt` → `{"text": ...}` JSONL → clean → dedup → merge |
| **Raw JSONL Stage** | `bhojpuri/data/archive_org_bhojpuri_raw/arch_bho_batch_*.jsonl` |
| **Cleaned JSONL Stage** | `bhojpuri/data/archive_org_bhojpuri_cleaned/cleaned_*.jsonl` (archived after merge) |
| **Rows After Cleaning** | [see config.json] |
| **Deduplication** | Against root accumulator |
| **State Tracking** | `archive_org_bhojpuri_state.json` (resumable; tracks downloaded item IDs and search cursor) |
| **IMPORTANT NOTE** | This collection is **automated via Bhojpuri-targeted search queries**, but items are filtered by archive.org's internal language tagging and subject metadata. Not all returned items may be authentic Bhojpuri (some could be mislabeled); quality varies. Cleaning pipeline (7-stage + 4-threshold validation) filters poor text automatically. |

#### 6. Manual Tesseract OCR of Bhojpuri Book PDFs

| Property | Value |
|----------|-------|
| **Source Name** | Manual OCR (Tesseract, scanned Bhojpuri books) |
| **Source Data** | Scanned PDF/image files of Bhojpuri books from archive.org or other sources |
| **License** | Variable (inherited from source books' original licenses) |
| **Collection Method** | **MANUAL STEP**: |
| | 1. User visits https://archive.org/search.php |
| | 2. Searches: `language:(Bhojpuri) AND mediatype:(texts)` |
| | 3. Filters to items with NO pre-OCR'd djvu.txt link (scanned-image-only) |
| | 4. Downloads PDF/image files and places in `bhojpuri/data_collect/ocr_sources/` |
| | 5. Runs `ocr_books_manual.py` |
| **OCR Process** | **Automated Script**: `ocr_books_manual.py` (uses existing `ocr_extractor.py`) |
| **Engine** | Tesseract 5.3.4 (system binary at `/usr/bin/tesseract`) |
| **Language Model** | `hin` (Hindi/Devanagari; same script as Bhojpuri) |
| **Image DPI** | 300 (configurable) |
| **Format** | PDF (via PyMuPDF rasterization) + PNG/JPG/TIFF images (direct OCR) |
| **Preprocessing** | Header/footer removal, hyphenation handling, newline collapsing (built into `ocr_extractor.py`) |
| **Pipeline** | Manual PDF curation → Tesseract OCR (via PyMuPDF) → `{"text": ...}` JSONL → clean → dedup → merge |
| **Raw JSONL Stage** | `bhojpuri/data/ocr_books_raw/batch_*.jsonl` |
| **Cleaned JSONL Stage** | `bhojpuri/data/ocr_books_cleaned/cleaned_*.jsonl` (archived after merge) |
| **Rows After Cleaning** | [see config.json] |
| **Deduplication** | Against root accumulator |
| **State Tracking** | `ocr_books_state.json` (tracks processed file SHA256 + mtime to skip re-processing) |
| **IMPORTANT NOTES** | |
| | - This is a **MANUAL curation step**: the archive.org API doesn't distinguish items with pre-OCR'd text from scanned-only PDFs. |
| | - OCR quality varies with image quality; some noise/artifacts are expected and handled by the cleaning pipeline. |
| | - Requires Python packages: `pytesseract>=0.3.10`, `PyMuPDF>=1.23.0` (installed via `pip install -r requirements.txt`). |
| | - System `tesseract` binary required: `sudo apt-get install tesseract-ocr tesseract-ocr-hin` (Hindi language pack). |
| | - Example books: "Bhojpuri Lok-geet", "Bhojpuri Vyakaran Aa Rachna", "Bhojpuri Bhasha aur Sahitya" (all found on archive.org). |

---

## Unified Processing Pipeline

Every source (whether automated or manual) follows the **same standardized chain**:

1. **Fetch/Extract Raw Data**
   - Convert to `{"text": ...}` JSONL format (one document per line, UTF-8)
   - Write to a dedicated `bhojpuri/data/<source>_raw/` directory

2. **Clean & Validate**
   - Apply `BhojpuriDataCleaner` (reused for all sources, thresholds unchanged)
   - **5-step text cleaning**: Unicode NFC normalization, citation removal, zero-width/control char stripping, Devanagari-only whitelisting, whitespace collapsing
   - **4-threshold validation**: Devanagari ratio ≥60%, min length ≥40 chars, min words ≥5, space density ≥0.05
   - Output: `bhojpuri/data/<source>_cleaned/cleaned_*.jsonl`

3. **Deduplicate Against Existing Corpus**
   - Load existing accumulator (`bhoj.txt`)
   - Check for exact MD5 matches and near-match MD5 (whitespace/punct normalized)
   - Filter new texts to include only those not already in corpus
   - Ensures no cross-source or source-accumulator duplicates

4. **Split & Merge Into Train/Val/Test**
   - Call `merge_ocr_into_splits()` with source-specific `cleaned_dir_name` and `temp_batch_filename`
   - Split the new batch: 80% train, 10% val, 10% test (seed 42, deterministic)
   - **Append** (never overwrite) the three split chunks onto existing `train/bhoj.txt`, `val/bhoj.txt`, `test/bhoj.txt`
   - Append the full new batch onto root accumulator (`bhoj.txt`)

5. **Update Configuration & Archive**
   - Run `update_config_ocr_fixed.update_config_bhojpuri()` to refresh:
     - `splits`: new train/val/test line counts
     - `token_progress`: total corpus tokens, % of 500M target
   - Append source name/URL to `config["data_sources"]` list
   - Archive intermediate `*_raw/` and `*_cleaned/` directories to prevent re-processing

---

## Deduplication & Quality Control

### Cleaning Thresholds (Applied to ALL sources identically)

| Stage | Method | Threshold/Action |
|-------|--------|-------------------|
| 1. Normalization | Unicode NFC | All text normalized to NFC form |
| 2. Citation Removal | Regex `\[N\]` | Removes Wikipedia-style citations |
| 3. Control Chars | Regex `[\x00-\x1f\x7f-\x9f]` | Strips zero-width, control chars |
| 4. Whitelisting | Devanagari `0x0900–0x097F` + digits + `.,"!?- ` | Drops non-Devanagari chars |
| 5. Whitespace | Collapse `\s+` → single space | Strip leading/trailing |
| **Validation** | | |
| Pre-strip Devanagari ratio | ≥60% | Reject if <60% Devanagari chars before cleaning |
| Min length | ≥40 chars | Reject if shorter |
| Min words | ≥5 words | Reject if fewer words |
| Space density | ≥0.05 | Reject if <5% spaces (very dense text likely garbage) |

### Deduplication Strategy

**Two-tier approach** (same as used for HF corpus in Phase 1):

1. **Exact Match**: MD5 hash of raw (pre-cleaned) text
   - Catches verbatim duplicates across sources
2. **Near-Match**: MD5 of text with whitespace/punctuation stripped + lowercased
   - Catches near-duplicates (minor formatting differences)

**Applied Against**: Existing root accumulator (`bhoj.txt`) + within-source dedup

**Reporting**: Each source logs rejection counts (exact duplicates + near duplicates removed)

---

## Configuration Updates & Progress Tracking

After each source is merged, the following fields in `bhojpuri/data/config.json` are **automatically updated**:

```json
{
  "data_sources": [
    "...original sources...",
    "bho.wikipedia.org (bhwiki-latest-pages-articles dump, ~8,857 articles)",
    "fossdot/bhojpuri-corpus (GitHub: reviewed community-contributed texts, ~171 sentences)",
    "BHLTR (shashwatup9k/bho-resources, monolingual corpus, CC BY-NC-SA 4.0)",
    "HuggingFace fineweb-2 (bho_Deva subset, ~18.7k CommonCrawl-derived docs)",
    "archive.org (Bhojpuri-tagged books, pre-extracted djvu.txt)",
    "Manual OCR (Tesseract, Bhojpuri book PDFs from archive.org/other sources)"
  ],
  "splits": {
    "train": {"lines": XXXXX, "percentage": 80},
    "val":   {"lines": XXXXX, "percentage": 10},
    "test":  {"lines": XXXXX, "percentage": 10}
  },
  "token_progress": {
    "total_corpus_tokens_estimate": XXXXX,
    "total_corpus_bytes": XXXXX,
    "target_tokens": 500000000,
    "progress_percent": XX.XX,
    "progress_str": "XXX.XM / 500M (XX.XX%)"
  }
}
```

---

## Execution Record

| Step | Source | Script | Status | Notes |
|------|--------|--------|--------|-------|
| 0 | N/A (Setup) | N/A | ✅ | Root `bhoj.txt` accumulator reconstructed from split fragments |
| 1 | Wikipedia | `download_wikipedia_dump.py` | ⏳ Pending | Authentic Bhojpuri Wikipedia (8,857 articles) |
| 2 | fossdot | `download_github_corpus.py` | ⏳ Pending | Community corpus (~171 reviewed sentences) |
| 3 | BHLTR | `download_bhltr_corpus.py` | ⏳ Pending | Monolingual + POS-annotated Bhojpuri |
| 4 | FineWeb-2 | `download_fineweb2_corpus.py` | ⏳ Pending | CommonCrawl-derived web text (~18.7k docs) |
| 5 | archive.org | `download_archive_org_bhojpuri.py` | ⏳ Pending | Pre-extracted djvu.txt from Bhojpuri-tagged books (~500 items target) |
| 6 | Manual OCR | `ocr_books_manual.py` | ⏳ Pending (Manual curation required) | Tesseract OCR of scanned Bhojpuri book PDFs |
| 7 | Final | `update_config_ocr_fixed.py` | ⏳ Pending | Update config.json with final stats |

---

## Expected Outcomes

After all sources are processed:

- **Corpus Size**: Estimate ~200–250M tokens (Phase 1 was 135.6M; these 6 sources are modest individually but accumulate)
- **Progress to 500M Target**: ~40–50% of target (still requires additional corpora beyond Phase 2, e.g., more OCR, parallel data, synthetic generation)
- **Data Authenticity**: Strong increase in genuine Bhojpuri text (Wikipedia dump is native-written, FineWeb-2/BHLTR/fossdot are validated Bhojpuri, OCR adds literary books)
- **Reduce Hindi Proxy Dependence**: Phase 1 relied heavily on Hindi Wikipedia as a Devanagari proxy; Phase 2 sources are authentic Bhojpuri

---

## References

- **Main Pipeline**: `bhojpuri/data_collect/` directory
- **Cleaning Config**: `BhojpuriDataCleaner` class in `data_cleaner.py`
- **Deduplication Function**: `deduplicate_with_existing_corpus()` in `download_hf_corpus.py`
- **Merge Logic**: `merge_ocr_into_splits()` in `ocr_merge.py`
- **Config Updates**: `update_config_bhojpuri()` in `update_config_ocr_fixed.py`
