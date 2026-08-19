# Synthetic Bhojpuri Generation — Quick Start Guide

Generated synthetic Bhojpuri text to grow the corpus toward the 500M token target. Runs as 3 independent background workers, resumable and isolated.

## Architecture

**3 parallel workers** (run simultaneously, each with its own state/directories):
- **Worker A**: Folk tales, cultural heritage, traditions (~15 topic prompts)
- **Worker B**: Village life, daily activities, family dialogue (~15 topic prompts)
- **Worker C**: Food, geography, proverbs, knowledge (~15 topic prompts)

Each worker:
- Generates Bhojpuri text via `claude -p` (Haiku model)
- Cleans with existing `BhojpuriDataCleaner` pipeline
- Deduplicates against corpus
- Merges into train/val/test splits (80/10/10)
- Updates `config.json` with `synthetic_generation` stats
- Resumes from state file if killed/restarted

## Files Created

- `bhojpuri/data_collect/synthetic_bhojpuri_generator.py` — Core orchestrator
- `bhojpuri/data_collect/run_synthetic_generation.py` — Launcher

## Usage

### Option 1: Single Worker (Foreground)
```bash
cd /media/ubuntu/Personal/IIIT\ Hyderabad/Semester\ 3/LMA/Mini_Project
python3 bhojpuri/data_collect/run_synthetic_generation.py \
  --worker-id a \
  --generations-per-cycle 50 \
  --cycle-delay 30
```

### Option 2: Single Worker (Background)
```bash
nohup python3 bhojpuri/data_collect/run_synthetic_generation.py \
  --worker-id a \
  --generations-per-cycle 50 \
  --cycle-delay 30 \
  > bhojpuri/data/synthetic_worker_a.log 2>&1 &
echo $! > bhojpuri/data/worker_a.pid
```

### Option 3: All 3 Workers in Parallel (Background) — RECOMMENDED
```bash
# Launch worker A
nohup python3 bhojpuri/data_collect/run_synthetic_generation.py \
  --worker-id a --generations-per-cycle 50 \
  > bhojpuri/data/synthetic_worker_a.log 2>&1 &
echo $! > bhojpuri/data/worker_a.pid

# Launch worker B
nohup python3 bhojpuri/data_collect/run_synthetic_generation.py \
  --worker-id b --generations-per-cycle 50 \
  > bhojpuri/data/synthetic_worker_b.log 2>&1 &
echo $! > bhojpuri/data/worker_b.pid

# Launch worker C
nohup python3 bhojpuri/data_collect/run_synthetic_generation.py \
  --worker-id c --generations-per-cycle 50 \
  > bhojpuri/data/synthetic_worker_c.log 2>&1 &
echo $! > bhojpuri/data/worker_c.pid
```

## Monitoring Progress

```bash
# View live status for a worker
cat bhojpuri/data/phase_synth_status_a.json | python3 -m json.tool

# Monitor progress log (real-time)
tail -f bhojpuri/data/phase_synth_progress_a.log

# Check all 3 workers at once
for w in a b c; do
  echo "=== Worker $w ===" 
  tail -3 bhojpuri/data/phase_synth_progress_$w.log
done

# View corpus growth in config
python3 -c "
import json
with open('bhojpuri/data/config.json') as f:
    c = json.load(f)
    print(f\"Tokens: {c['token_progress']['progress_str']}\")
    if 'synthetic_generation' in c:
        for w, stats in c['synthetic_generation'].items():
            print(f\"{w}: {stats['merged_lines']} lines merged\")
"
```

## Stopping Workers

```bash
# Graceful stop (SIGTERM)
kill -TERM $(cat bhojpuri/data/worker_a.pid)

# Kill immediately (SIGKILL)
kill -9 $(cat bhojpuri/data/worker_a.pid)

# Stop all 3
for w in a b c; do kill -TERM $(cat bhojpuri/data/worker_$w.pid) 2>/dev/null; done
```

## Resuming After Interruption

Workers automatically resume from saved state on restart:
```bash
# Just re-run the command — it will load the state file and continue
nohup python3 bhojpuri/data_collect/run_synthetic_generation.py \
  --worker-id a --generations-per-cycle 50 \
  > bhojpuri/data/synthetic_worker_a.log 2>&1 &
```

State is saved in:
- `bhojpuri/data/synthetic_gen_state_a.json` (batch IDs, attempt counts, line counts)
- `bhojpuri/data/phase_synth_progress_a.log` (cycle-by-cycle log)
- `bhojpuri/data/phase_synth_status_a.json` (live status snapshot)

## Expected Performance

**Throughput**: ~4 passages/min per worker (~15 seconds per `claude -p` call)

**Realistic timeline** (to close 363M token gap):
- 1 worker: ~250 days continuous
- 3 workers in parallel: ~84 days continuous
- Over 4 weeks: realistic gain ~40-100M tokens (partial progress)

This is **background work** — run continuously while you proceed with Phase 2 model training.

## Troubleshooting

**No output in log?**
- Check if process is running: `ps aux | grep run_synthetic_generation`
- Check stderr: `tail -f bhojpuri/data/synthetic_worker_a.log`
- Check state file exists: `ls bhojpuri/data/synthetic_gen_state_a.json`

**"No text generated this cycle"?**
- Claude CLI may be returning empty responses
- Check `claude -p` manually: `claude -p "भोजपुरी में कहानी सुनाइए।" --model haiku`
- Check internet connectivity / Claude service availability

**Corpus not growing?**
- Generated text may all be rejected by Bhojpuri marker check or cleaning pipeline
- Check merged lines in `phase_synth_status_a.json` (should be > 0 after each cycle)
- Manually inspect generated files: `cat bhojpuri/data/synthetic_a_raw/synthetic_batch_000000.jsonl | head -1 | python3 -m json.tool`

## Configuration

Adjust per-worker settings in launcher:
```bash
--generations-per-cycle 50      # Texts to generate per cycle (default: 50)
--cycle-delay 30                # Seconds between cycles (default: 30)
--target-tokens 500000000       # Target corpus size (default: 500M)
```

Larger `--generations-per-cycle` = longer cycles but better dedup/merge amortization.
Smaller `--cycle-delay` = faster iteration but less time between cycles for inspection.

## Next Steps

1. **Launch 3 workers** in parallel (see "All 3 Workers in Parallel" above)
2. **Monitor progress** via logs/status files every few hours
3. **Proceed with Phase 2** (model training) while workers run in background
4. **Check corpus growth** periodically; relaunch workers if killed

Total corpus growth expected by deadline (16 Sep 2026): ~40-100M tokens added to existing 136.6M.
