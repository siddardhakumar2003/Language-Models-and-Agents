# Phase 1b Quick Start Guide

## What Was Built

✅ **Archive.org Book Downloaders**
- `telugu/data_collect/archive_org_downloader.py` — Downloads pre-OCR'd Telugu books
- `bhojpuri/data_collect/archive_org_downloader.py` — Downloads pre-OCR'd Hindi books (Bhojpuri proxy)

✅ **News Article Crawlers**
- `telugu/data_collect/news_crawler.py` — Crawls 10 Telugu news sites
- `bhojpuri/data_collect/news_crawler.py` — Crawls 10 Hindi news sites

✅ **Orchestrators**
- `telugu/data_collect/run_phase1b.py` — Main continuous loop for Telugu
- `bhojpuri/data_collect/run_phase1b.py` — Main continuous loop for Bhojpuri

✅ **Config Update (Fixed)**
- `telugu/data_collect/update_config_ocr_fixed.py` — Reads from cumulative stats
- `bhojpuri/data_collect/update_config_ocr_fixed.py` — Reads from cumulative stats

## Quick Test (Small Scale)

Test the full pipeline with a tiny data cap:

```bash
cd "/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project"

venv/bin/python3 -m telugu.data_collect.run_phase1b \
    --target-tokens 100000 --books-per-cycle 3 --articles-per-cycle 10

# Watch progress
tail -f telugu/data/phase1b_progress.log
```

Takes ~5-10 min. Verify:
- Archive.org downloader works
- News crawler finds articles
- Data cleaner processes them
- Merge appends safely
- Config updates correctly

## Full 500M Token Run

```bash
cd "/media/ubuntu/Personal/IIIT Hyderabad/Semester 3/LMA/Mini_Project"

setsid nohup venv/bin/python3 -m telugu.data_collect.run_phase1b \
    --target-tokens 500000000 \
    --books-per-cycle 100 \
    --articles-per-cycle 500 \
    > telugu/data/phase1b_nohup.out 2>&1 < /dev/null &
disown
echo $! > telugu/data/phase1b.pid

# Monitor
tail -f telugu/data/phase1b_progress.log

# Similarly for Bhojpuri
setsid nohup venv/bin/python3 -m bhojpuri.data_collect.run_phase1b \
    --target-tokens 500000000 \
    --books-per-cycle 100 \
    --articles-per-cycle 500 \
    > bhojpuri/data/phase1b_nohup.out 2>&1 < /dev/null &
disown
echo $! > bhojpuri/data/phase1b.pid
```

## Stop/Resume

```bash
kill $(cat telugu/data/phase1b.pid)
kill $(cat bhojpuri/data/phase1b.pid)

# Resume: just re-run the same command
```

All state is persisted — no data loss or duplication.

## Progress Log

Each cycle logged to `telugu/data/phase1b_progress.log`:
```
[2026-08-14T10:23:45] CYCLE 5 | books:+12(18.4K) | articles:+340(6.1K) | cleaned: kept=14220 rejected=4502 (75.9%) | corpus: 187.4M/500M (37.5%) | elapsed:125.3s
[2026-08-15T14:11:02] STOPPED reason=target_reached corpus=500.2M/500M(100%) elapsed=28h47m
```

OR if sources exhaust before target:
```
[2026-08-20T09:15:33] STOPPED reason=sources_exhausted corpus=298.1M/500M(59.6%) -- TARGET NOT REACHED
```

## Key Design Points

- **`te.txt` untouched** — Never modified by any new code
- **Append-only splits** — `train/val/test/telugu.txt` and `bhoj.txt` grow incrementally
- **Crash-safe & resumable** — All state in JSON files, idempotent downloads/crawls
- **Network-bound** — Expected: 12-30h for Bhojpuri, 2-7+ days for Telugu (sources may run short)
- **Honest progress** — Stops gracefully if sources exhaust; logs true achieved tokens

## Verify Output

```bash
# Check line/size growth
wc -l telugu/data/telugu.txt telugu/data/train/telugu.txt telugu/data/val/telugu.txt telugu/data/test/telugu.txt

# Check final config
cat telugu/data/config.json | jq '.token_progress'

# First and last log lines
head -5 telugu/data/phase1b_progress.log
tail -5 telugu/data/phase1b_progress.log
```

---

**Start with the small test first!** ✓
