# Phase 3 Report: Reasoning Finetuning, Attention Analysis, and Final Comparison

**Status note (read first):** The reasoning-finetuning numbers in Sec. 1 below are from the
*pre-fix* finetuning run (5 epochs, uniform-mode data generation, plain cross-entropy loss).
That run collapsed (Sec. 1.3). The fixes described there (rebalanced data generator, weighted
loss, 20 epochs) are implemented and committed but **not yet re-run on Kaggle** as of this
report. Sec. 1.4 gives the expected shape of the fix; the actual post-fix numbers must replace
the placeholders there before final submission -- re-run both `finetune_kaggle.ipynb` notebooks
and re-run `report/phase-3/code/attention_analysis.py` to refresh Sec. 2's attention comparison
against the new finetuned checkpoints.

---

## 1. Reasoning Finetuning (PDF Sec. 3.1)

### 1.1 Protocol

Both models were finetuned independently, starting from each language's own Phase 2 pretrained
checkpoint (tokenizer/vocabulary kept fixed, full-parameter finetuning -- no adapters, no new
heads, no frozen layers). Loss is cross-entropy masked to answer tokens only
(`IGNORE_INDEX=-100` on the prompt). See `telugu/train/kaggle_bundle/finetune/finetune.py` /
`bhojpuri/.../finetune.py` and `{telugu,bhojpuri}/configs/finetune_config.json`.

### 1.2 Synthetic reasoning dataset

Generated programmatically (`{telugu,bhojpuri}/finetune/generate_reasoning_data.py`), covering
five question types in the target language's script:

- **pairwise_value** -- "A's height is X, B's is Y, who is taller?" (answer: a name)
- **pairwise_yesno** -- "Is A taller than B?" (answer: yes/no)
- **transitive** (max / min / yesno) -- A>B, B>C given, asks the extreme or an A-vs-C yes/no
- **three_value_superlative** (max / min) -- three entities with explicit values
- **equal** -- two entities with the same value

10,000 examples (8000/1000/1000 train/val/test), over height/weight/age (person entities) and
price (object entities). **Leakage avoidance:** person and object name pools are shuffled and
split 70/15/15 into train/val/test *before* generation, so no name seen in val/test ever
appears in train; additionally one phrasing variant per question type (`_ho` suffix) is
reserved exclusively for val/test, holding out a relation-pattern as well as entity identity.
Bhojpuri template phrasing is best-effort and not verified by a native speaker (noted in
`bhojpuri/finetune/data/stats.json`).

### 1.3 First run: mode collapse (diagnosed and fixed)

The first finetuning run (5 epochs, `finetune_config.json` defaults) produced:

| | Telugu (H) | Bhojpuri (L) |
|---|---|---|
| Pretrained (zero-shot) test accuracy | 0.0 | 0.0 |
| Finetuned test accuracy (best val-loss ckpt) | 27.6% | 32.8% |
| Best val PPL | 44.3 (epoch 2) | 33.1 (epoch 1) |

The aggregate accuracy looked plausible, but breaking it down by question type exposed the
real behavior:

| Question type | Telugu acc. | Bhojpuri acc. |
|---|---|---|
| equal | 83.96% / 100%\* | 100% |
| pairwise_value | **0%** | **0%** |
| pairwise_yesno | 36.6% / 45.5%\* | 45.5% |
| three_superlative_max/min | **0%** | **0%** |
| transitive_max/min | **0%** | **0%** |
| transitive_yesno | 90.9% / 100%\* | 100% |

\*Two Telugu runs are shown (20-epoch pre-rebalance run / earlier 5-epoch run); the pattern is
identical in both.

**Diagnosis:** the model wasn't failing to generalize to unseen entity names -- it was
outputting the *same fixed token* (`సమానం`/`బరాబర్` "equal" or `అవును`/`हँ` "yes") regardless of
the question, even getting plain yes/no questions wrong (`gold='కాదు' pred='సమానం'`). Root
cause: `equal` always answers one fixed token regardless of entities, and `transitive`'s
`yesno` sub-case is *always* "yes" by construction (A>B>C implies A>C). Uniform mode sampling
let these two answer-token sinks dominate ~27% of training examples, while any individual
entity name (needed for the other four question types) was spread thin over ~35-50 names.
Standard cross-entropy took the frequency shortcut instead of learning to compare values or
copy names.

### 1.4 Fix (implemented, not yet re-run)

Three changes, all committed:

1. **Rebalanced the data generator's mode weights** (`generate_reasoning_data.py`):
   `equal` down-weighted 20%→5%, `transitive`'s `yesno` sub-case 33%→20%, reasoning-required
   modes up-weighted correspondingly. Verified on regenerated data: `equal`'s answer token
   dropped from ~20% to ~5% of training examples.
2. **Widened the entity pool**: 50→100 person names, 20→40 object names (both languages), so
   held-out test names have more train-side analogues and no single name dominates.
3. **Inverse-frequency class-weighted loss** (`compute_answer_token_weights` in `finetune.py`):
   computes per-token weight from the train set's actual answer-token distribution and passes
   it to `nn.CrossEntropyLoss(weight=...)`. Verified locally: `సమానం`/`అవును`/`కాదు` weighted
   down to 0.07-0.2x, individual names weighted up to ~1.2-2.8x.
4. `num_epochs` raised 5→20, now safe against overfitting because the final reported accuracy
   is taken from `checkpoint_best.pt` (lowest val loss), not the last epoch (a related bug
   fixed alongside this: the original script scored the *final*-epoch weights even though val
   loss consistently bottomed out around epoch 1-2 of 5).

**TODO before submission:** re-run both `finetune_kaggle.ipynb` notebooks (rebalanced data +
weighted loss + 20 epochs), then re-run the per-question-type breakdown cell added to each
notebook, and replace the table in Sec. 1.3 / this section with the post-fix numbers.

### 1.5 Qualitative examples (pre-fix run)

Failure mode was uniform across question types requiring a name answer:

```
ప్రశ్న: రాజు ఎత్తు 178 సెం.మీ. సూర్య ఎత్తు 155 సెం.మీ. ఎవరు ఎక్కువ ఎత్తు కలిగి ఉన్నారు?
gold='రాజు'  pred='సమానం'

सवाल: गीतांजलि: 18 बरिस, सीता: 59 बरिस, सावित्री: 42 बरिस (उमिर के हिसाब से)। इनमें सभसे कम...
gold='गीतांजलि'  pred='बराबर'
```

Successes were concentrated in the trivially-answerable `equal`/`transitive_yesno` categories,
which is exactly what the diagnosis in 1.3 predicts.

### 1.6 Model H vs. Model L on reasoning

Pre-fix, both models show the *identical* failure pattern (same question types at 0%, same
qualitative collapse) -- the resource-tier gap did not show up as a difference in reasoning
*strategy*, only in raw exact-match numbers (Bhojpuri's 32.8% vs. Telugu's 27.6%, likely
reflecting Bhojpuri's better-converged pretrained base at the time: PPL 814 vs. Telugu's
in-progress PPL ~4120 -- see Sec. 3). This should be re-examined once both models are re-run
with the fix, since a genuinely-learned copy/compare mechanism (rather than a frequency
shortcut) is a much more informative point of H-vs-L comparison than the pre-fix numbers.

---

## 2. Attention Analysis (PDF Sec. 3.2, post-finetune)

**Methodology note:** `generate_plots.py` (used for the Phase 2 report) used to fabricate its
attention heatmaps with `np.random.randn` plus a hand-coded "local attention bias" multiplier
-- it never called the model. That has since been fixed: `report/phase-2/code/real_attention_analysis.py`
now regenerates `report/phase-2/plots/attention_complete/` from real forward passes on the
Phase 2 submission checkpoints, and `report/phase-2/report.md` Sec 6.4 has been corrected to
match (see that section's "Correction note"). Phase 3's analysis below was never built on the
fake generator; it independently runs real forward passes
(`model(input_ids, return_attn=True)`) via `report/phase-3/code/attention_analysis.py`.

### 2.1 Setup

For each language: loaded the pretrained checkpoint and the best finetuned checkpoint
(currently the pre-fix one, per the Sec. 1 status note), ran both on the same real
comparative-reasoning test prompt, and plotted every head's attention at layer 0 (early) and
the last layer (Telugu: layer 9 of 10; Bhojpuri: layer 7 of 8) -- see
`report/phase-3/plots/attention/*.png`. Entropy and mean attention distance were computed per
head/layer, averaged over 40 held-out test prompts, for both pretrained and finetuned.

Example prompt used for the heatmaps:
- Telugu: *"ప్రశ్న: రవి, దీప ఇద్దరి వయస్సు 13 సంవత్సరాలు చొప్పున సమానంగా ఉంటే, ఎవరిది ఎక్కువ?"* (gold: సమానం)
- Bhojpuri: *"सवाल: पूजा के उमिर 72 बरिस अवुरी बाबू के उमिर 10 बरिस बा। इनमें उमिर में जादा के बा?"* (gold: पूजा)

### 2.2 What the heatmaps show

All heads at layer 0 (both languages, both pretrained and finetuned) show a strong **attention
sink**: every query position attends heavily to position 0 (the first token), with a smaller
amount of mass on nearby preceding tokens and almost nothing further back -- a well-documented
pattern in transformer language models, not an artifact. The causal mask is visibly respected
(strict lower-triangular structure, zero weight above the diagonal) in every plot, which is
also a direct empirical check that causal masking is implemented correctly (PDF's "verify
empirically that the model cannot see the future" requirement).

### 2.3 Entropy / mean attention distance (real numbers)

| Layer | Telugu entropy (pre / fine) | Telugu distance (pre / fine) | Bhojpuri entropy (pre / fine) | Bhojpuri distance (pre / fine) |
|---|---|---|---|---|
| 0 (early) | 2.470 / 2.469 | 7.163 / 7.163 | 2.425 / 2.424 | 7.198 / 7.197 |
| 1 | 2.334 / 2.279 | 6.996 / 7.121 | 2.230 / 2.169 | 5.897 / 5.648 |
| 2 | 2.323 / 2.273 | 7.055 / 7.063 | 2.159 / 2.299 | 6.876 / 6.899 |
| 3 | 2.465 / 2.468 | 7.268 / 7.189 | 2.335 / 2.440 | 7.410 / 7.475 |
| mid-late (avg) | ~2.47 / ~2.47 | ~7.2 / ~7.18 | ~2.50 / ~2.50 | ~7.57 / ~7.60 |
| last | 2.471 / 2.471 | 7.185 / 7.178 | 2.518 / 2.515 | 7.612 / 7.643 |

(Full per-head numbers: `report/phase-3/metrics/{telugu,bhojpuri}_attention_summary.json`.)

**Local vs. long-range / head specialization:** layers 1-2 are the most locally-specialized in
*both* models -- lowest entropy and shortest mean distance of any layer (most pronounced in
Bhojpuri, where layer 1's mean distance drops to ~5.6-5.9 vs. ~7.2-7.6 everywhere else). Layer
0 and layers 3-onward stay close to maximum entropy for the sequence length and barely
differ from each other, i.e. **most of the depth of both models has not developed strong
per-layer specialization** -- consistent with both models still being comparatively
undertrained (Telugu PPL ~4120, Bhojpuri PPL ~814; see Sec. 3) relative to a fully converged
LM, where sharper, more differentiated per-layer attention patterns are typical.

**Pretrained → finetuned shift (pre-fix checkpoint):** changes are small in absolute terms
(entropy deltas of 0.01-0.17, distance deltas of 0.01-0.25), expected given the tiny learning
rate (2e-5) and how few finetuning steps the pre-fix run actually took before its best
checkpoint (step 500-1000). The one consistent, non-trivial shift is layer 1 becoming *more*
local after finetuning (Bhojpuri: distance 5.897→5.648; Telugu: entropy 2.334→2.279) --
plausible given the QA templates place the numeric comparison very close to the answer
position, so the task rewards sharpening exactly that layer's local attention. This should be
re-measured against the post-fix finetuned checkpoint, where genuine task learning (rather than
frequency-shortcut convergence) may produce a larger, more meaningful shift.

---

## 3. Model H vs. Model L: Data, Pretraining, and Resource-Tier Comparison

| | Telugu (Model H) | Bhojpuri (Model L) |
|---|---|---|
| Vocabulary | 20,000 (WordPiece) | 16,000 (WordPiece) |
| Architecture | 10 layers, d_model=384, 8 heads, 25.5M params | 8 layers, d_model=320, 8 heads, 15.1M params |
| Pretraining data | ~509M-token target corpus | ~292M-token target corpus (lower-resource) |
| Pretrained PPL (current checkpoint) | **~4120** (epoch 36, still training, log last updated 2026-09-13) | **814.5** (epoch 44, converged/stopped 2026-09-10) |

**Two Telugu checkpoints exist, used deliberately side by side (not one "canonical" choice):**
an earlier, smaller, architecture-matched pair (`telugu/model/outputs/submission/`,
`bhojpuri/model/outputs/checkpoints_sub/` -- 6 layers, 10K vocab, 7.34M params each,
deliberately sized identically for a clean H-vs-L comparison), and the larger,
per-language-sized architecture the project continued training past that point (25.5M params,
20K vocab, still training as of this report). Both are reported here rather than picking one,
because each is better on a different axis:

| | Submission (7.34M, 6 layers) | Current large (25.5M, 10 layers) |
|---|---|---|
| Validation PPL | **881.9** (converged, epoch 16) | ~4120 (still improving, epoch 36) -- 4.7x worse |
| Training steps at this checkpoint | 3.34M | 855K (4x fewer) |
| Attention (real, Sec 2 methodology) | Flat attention-sink pattern, **identical entropy AND std across all 6 layers** (2.317±0.398 everywhere) -- no per-layer differentiation at all | Same attention-sink-dominant pattern through most layers (entropy ≈2.47), but a handful of heads in layers 1-2 break away to markedly lower entropy (1.92-2.16) -- a small amount of real per-head specialization the submission checkpoint doesn't show |

Neither checkpoint shows the textbook "early=broad, late=focused" story -- both lean heavily on
an attention-sink shortcut through most of their depth, consistent with both being
comparatively undertrained relative to a fully converged LM. The honest reading: **the
submission checkpoint is the better language model by a wide margin (PPL), but shows zero
attention differentiation across depth; the larger checkpoint has a small amount of emergent
head specialization despite far fewer training steps, at a steep PPL cost.** This tradeoff --
not a single "Telugu's checkpoint was PPL X" number -- is what should be reported for any
H-vs-L comparison below. Re-check `telugu/model/outputs/checkpoints/training_telugu.log`'s
latest lines before finalizing, since the large checkpoint is still training.

### 3.1 Required discussion questions (PDF Sec. 3.3)

**1. How did data scale and quality differ between Model H and Model L?**
Telugu targeted ~509M tokens vs. Bhojpuri's ~292M -- consistent with Bhojpuri being the
designated lower-resource language. Both corpora meet the ≥20% manual-collection requirement
from Phase 1. Despite the token-count gap favoring Telugu, on the matched (submission)
architecture both languages converge to essentially the same PPL (881.9 vs. 870.1) -- i.e. at
that model size, the data-scale gap did not translate into a language-modeling quality gap.
The larger Telugu-only architecture (Sec. 3 table) has not yet demonstrated it can use its
extra capacity/data advantage effectively (PPL ~4120, still training).

**2. How do language-modeling and reasoning results compare across the two resource tiers?**
On the architecture-matched checkpoints: near-identical PPL (881.9 Telugu vs. 870.1 Bhojpuri)
and a shared qualitative attention failure mode (flat, undifferentiated attention-sink pattern
in both, Sec. 3 table) -- the resource-tier gap essentially doesn't show up here. On reasoning
(pre-fix finetuning, run from the larger, currently-underperforming Telugu checkpoint rather
than the submission one): near-identical failure pattern in both languages (same question
types at 0%), with Bhojpuri's 32.8% aggregate exact-match slightly ahead of Telugu's 27.6%.
Because the two languages' reasoning runs started from checkpoints on different points of the
PPL/attention tradeoff (Sec. 3), this 27.6-vs-32.8 gap is not a clean resource-tier
comparison and should be re-read once finetuning is re-run post-fix.

**3. What tokenizer/corpus factors most affected the lower-resource model?**
Bhojpuri's smaller vocabulary (16K vs. 20K) and person/object name pool (widened from 50/20 to
100/40 in the finetuning-data fix, same scale as Telugu) were kept close to Telugu's to isolate
the resource-tier effect rather than compound it with tokenizer-design differences. The
finetuning-data mode-collapse (Sec. 1.3) affected both languages identically, so it is not a
resource-tier-specific factor -- it was a data-generation bug independent of language.

**4. What evidence explains the observed differences?**
The reasoning-finetuning collapse is fully explained by answer-token frequency imbalance in
the synthetic dataset (Sec. 1.3) -- verified directly by inspecting per-example model outputs,
not inferred. The Telugu-vs-Bhojpuri PPL/attention tradeoff (Sec. 3) is explained by the two
Telugu checkpoints occupying different points on a training-steps-vs-architecture-size
tradeoff (3.34M steps on a 7.34M-param model vs. 855K steps on a 25.5M-param model) -- verified
directly from each checkpoint's embedded step count and the real attention entropy/std
computed per layer, not inferred from PPL alone.

---

## 4. Ablation Study: Telugu Low-Parameter vs. High-Parameter Model

Both variants analyzed with real per-head attention extraction (no fabricated data --
`report/phase-2/code/real_attention_analysis.py`), all heads, every layer.

### 4.1 Setup

| | Low-parameter model | High-parameter model |
|---|---|---|
| Checkpoint | `telugu/model/outputs/submission/checkpoint_best.pt` | `telugu/model/outputs/checkpoints/checkpoint_best.pt` |
| Layers | 6 | 10 |
| d_model | 256 | 384 |
| Vocab | 10,000 | 20,000 |
| Parameters | 7,341,840 | 25,543,712 |
| Training steps at this checkpoint | 3,342,524 | 855,272 |
| Validation PPL | **881.90** | **4119.81** |
| Heatmaps | `report/phase-2/plots/attention_complete/telugu/low_parameter_model/` | `.../telugu/high_parameter_model/` |

Both were run on the same 40 real held-out Telugu test sentences (`telugu/data/test/telugu.txt`,
not synthetic/random text), with the low-parameter model using the original 10K-vocab tokenizer
it was actually trained with (recovered from the `SidLMA/` mirror -- the current tokenizer was
later retrained to 20K vocab and is incompatible with this checkpoint's embedding table) and the
high-parameter model using the current 20K tokenizer.

### 4.2 Per-layer entropy and mean attention distance (real, all heads averaged)

| Layer | Low-param entropy | Low-param distance | High-param entropy | High-param distance |
|---|---|---|---|---|
| 0 | 2.317 | 6.537 | 2.198 | 5.743 |
| 1 | 2.317 | 6.537 | 2.104 | 5.708 |
| 2 | 2.317 | 6.537 | 2.055 | 5.664 |
| 3 | 2.317 | 6.537 | 2.191 | 5.772 |
| 4 | 2.317 | 6.537 | 2.195 | 5.776 |
| 5 | 2.317 | 6.537 | 2.197 | 5.774 |
| 6 | -- (only 6 layers) | -- | 2.198 | 5.754 |
| 7 | -- | -- | 2.198 | 5.757 |
| 8 | -- | -- | 2.199 | 5.751 |
| 9 | -- | -- | 2.199 | 5.752 |

### 4.3 Findings

1. **The low-parameter model's attention is completely flat across depth** -- entropy and
   distance identical to 3 decimal places on every one of its 6 layers (2.317 / 6.537
   everywhere). Every layer converged to the same attention-sink-plus-local-decay pattern; there
   is no early-vs-late differentiation to report.
2. **The high-parameter model shows small but real per-layer variation** -- entropy dips at
   layers 1-2 (2.104, 2.055) relative to the other 8 layers (~2.19-2.20), and the same dip shows
   up in distance (5.708, 5.664 vs. ~5.75-5.78 elsewhere). This is a genuine, if modest, sign of
   depth-wise specialization the low-parameter model doesn't show at all -- achieved in 4x fewer
   training steps (855K vs. 3.34M).
3. **The tradeoff is real and goes in opposite directions on the two axes that matter**:
   language-modeling quality (PPL) strongly favors the low-parameter model (4.7x better);
   attention differentiation, the one structural signal available from this analysis, mildly
   favors the high-parameter model. Neither model develops the clean "local heads vs. long-range
   heads" specialization a fully-converged transformer typically shows -- both are still
   comparatively undertrained language models by that standard.
4. **What this predicts for finetuning (pending, see below):** since PPL is the standard proxy
   for how much usable linguistic structure a base model has to adapt from, the low-parameter
   model is the more likely candidate to finetune well despite its smaller capacity -- worth
   testing directly rather than assuming from architecture size alone.

### 4.4 Finetuning comparison (setup complete, run pending)

A parallel finetuning pipeline for the low-parameter checkpoint has been built:
`telugu/finetune/telugu_finetune_low.ipynb`, `telugu/configs/finetune_low_config.json` (and its
`kaggle_bundle` copy), and a matching `tokenizer_config_low.json` +
`telugu_wp_tokenizer_low.json` -- all new files, `finetune_config.json` and the normal
`finetune_kaggle.ipynb`/checkpoints are untouched. Every hyperparameter in
`finetune_low_config.json` is identical to `finetune_config.json` (same LR, epochs, batch size,
schedule) so that architecture/checkpoint choice is the only variable between the two runs --
`finetune.py` was extended with optional `model_architecture` / `tokenizer_filename` /
`tokenizer_config_filename` config fields (only used when present, so the normal run's code path
is unchanged) to make this possible without duplicating the finetuning script.

**Still needed:** upload `telugu/model/outputs/submission/checkpoint_best.pt` as a new Kaggle
dataset (e.g. `checkpoint-telugu-low`), attach it to `telugu_finetune_low.ipynb` along with the
re-exported `kaggle_bundle`, and run it. Its final cell produces the same per-question-type
accuracy breakdown as the normal notebook (`test_eval_breakdown.json`, tagged
`"model_variant": "low_parameter"`) for direct comparison against Sec. 1.3's numbers.

---

## 5. Reproducibility

- Reasoning data generation: `python3 {telugu,bhojpuri}/finetune/generate_reasoning_data.py --num-samples 10000 --seed 42`
- Finetuning: `{telugu,bhojpuri}/finetune/finetune_kaggle.ipynb` on Kaggle, `ROOT_DIR` pointing
  at the corresponding `kaggle_bundle/` dataset, `PRETRAINED_CKPT` pointing at that language's
  Phase 2 pretrained checkpoint.
- Attention analysis: `python3 report/phase-3/code/attention_analysis.py --language both`
  (reads checkpoints directly from `{telugu,bhojpuri}/model/outputs/checkpoints/` and
  `{telugu,bhojpuri}/finetune/outputs/finetune_checkpoints/`; no GPU required, runs on CPU).
- Telugu low-vs-high-parameter ablation (Sec. 4): pretrained comparison via
  `python3 report/phase-2/code/real_attention_analysis.py` (also CPU, no GPU required);
  finetuning comparison via `telugu/finetune/telugu_finetune_low.ipynb` on Kaggle (Sec. 4.4).

### Outstanding items before final submission

- [ ] Re-run both finetuning notebooks with the rebalanced data + weighted loss; replace Sec.
      1.3/1.6 numbers with post-fix results.
- [ ] Re-run `attention_analysis.py` against the new finetuned checkpoints; replace Sec. 2.3.
- [ ] Run `telugu_finetune_low.ipynb` (Sec. 4.4) once the low-parameter checkpoint is uploaded
      to Kaggle as a new input dataset; fill in the finetuning half of the ablation study.
- [ ] Resolve the Telugu pretrained-checkpoint situation (Sec. 3 caveat) -- either let the
      current retrain converge further, or revert to the architecture-matched submission
      checkpoint, and update Sec. 3's table accordingly. Sec. 4's ablation results are exactly
      the evidence needed to make this call once Sec. 4.4 is filled in.
- [ ] Upload finetuned + pretrained checkpoints to Google Drive (both Telugu variants once
      Sec. 4.4 is run); add shareable links to `README.md` (required by the PDF's submission
      checklist -- not yet done for any checkpoint as of this report).
