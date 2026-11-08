# remaining_checks.md

Answers to the seven items, with the file that settles each. Anything not supplied is
marked **not available** or **not yet run** rather than glossed.

All tables are written at full numeric precision, not rounded to manuscript values.

---

## 1. Preprocessing fix: which outputs were regenerated

The leak was that `mu, sd` were fitted on the full matrix before the train/test split,
in three scripts. All three now fit on the training split only.

| Output | File | Regenerated after the fix? | Before/after |
|---|---|---|---|
| 476-speaker classification | `U_classification_476_speakers.tsv` | **yes** | `U_classification_476_speakers.PRE_FIX.tsv` |
| Classification layer-selection scores | same file (`clf_picked_layer` column) | **yes** | same PRE_FIX file |
| Development depth curves (figure with the layer curves) | `V_layer_curves.tsv` | **yes** | `V_layer_curves.PRE_FIX.tsv` |
| Cross-corpus classification | `T_layer_transfer_classification.tsv` | **yes** | `T_layer_transfer_classification.PRE_FIX.tsv` |

Two of these (`V`, `T`) were **stale when you asked** — the scripts had been corrected
but not re-run. Both have now been regenerated.

Measured effect of the fix:

- `U`: largest change in selected-layer accuracy **0.0003**; no layer choice moved.
- `T`: accuracies identical to 4 decimal places; Whisper's selected layer moved
  12 → 13, which now agrees with the clustering-selected layer.
- `V`: largest change **0.00224** across 158 rows, mean 0.00034; **every peak-layer
  choice identical**.

---

## 2. Seed and clustering settings

Full table: `S17_clustering_settings.tsv`.

| Table | Seeds | n_init | batch_size | max_iter | Evaluated on |
|---|---|---|---|---|---|
| `J_layer_transfer` | (0,) | 3 | 1024 | sklearn default (100) | OpenSLR-30 2,064 utts / 12 spk; content on OpenSLR-52 speakers B, 2,229 utts / 238 spk |
| `Q_layer_seeds_uniform` | (0,1,2,3,4) | 5 | 1024 | 100 (pinned, `tol=0.0`, `max_no_improvement=10`) | OpenSLR-30 2,064 utts / 12 spk |
| `O_layer_transfer_omnilingual` | **(0,1,2,3,4) — five, not three** | 5 | 512 | sklearn default (100) | Omnilingual 415 utts / 7 spk |
| `P_mfcc_vs_layer` | (0,1,2) | 5 | 512 | sklearn default (100) | OpenSLR-30 2,064 / 12 spk; OpenSLR-52 11,155 / 476 spk |

**Omnilingual (table O) uses five seeds.** An earlier appendix said three; that was
wrong. The generating script is `eval/layer_transfer_omni_eval.py`, promoted out of
`/tmp` so the table no longer depends on a temp file.

**Matched settings within each comparison:**

- `Q` — yes. Selected, final, all-layer-mean and the test-chosen oracle all use five
  seeds. The oracle previously used one seed, which made it incomparable; corrected.
- `O` — yes, all four conditions at five seeds.
- `J` — yes, single seed throughout.
- `P` — **no, and this is a known mismatch.** The MFCC rows use `n_init=5`,
  `batch_size=512`; the encoder rows in the same table are carried over from `J`
  (`n_init=3`, `batch_size=1024`). The MFCC-versus-encoder ratio in that table is
  therefore across slightly different k-means settings. The effect is expected to be
  small relative to the 0.55× / 2.00× gap it reports, but it has **not been measured**.

Development and target settings differ between `J` (selection) and `Q`/`O` (scoring).
That is documented rather than reconciled, per your note.

---

## 3. Reranking counts

Summary: `S18_rerank_counts.tsv`. Per-row metadata: `S14_rerank_query_ids.tsv`.

| Quantity | Value |
|---|---|
| Requested queries | 500 (`--rerank-queries` default) |
| Generated queries | **500** |
| Unique prompts in the dataset | 1,129 |
| Candidate recordings | 2,384 (all rows serve as candidates) |
| Queries excluded for lacking a cross-speaker positive | **0** |
| Final scored queries | **500** |

How the figures relate:

- **2,384** is the row count of the prepared dataset. Every row is a candidate.
- **500** of those rows carry a `query_id`. They were selected at preparation time to
  have at least one cross-speaker positive, so none is excluded at evaluation.
  Positives per query: min 1, median 2, max 6.
- **1,129** is the number of distinct prompts among all 2,384 rows, of which 282 have
  only one speaker. Those 282 prompts appear **only as candidates and never as
  queries**, so they exclude nothing.
- **847** was my arithmetic of 1,129 − 282, presented as an effective query count.
  That subtracted the prompt inventory from itself and has no referent in the
  evaluated data. **Withdrawn.**

No figure here belongs to an older dataset version; all are from the evaluated
`openslr52_rerank` on disk.

---

## 4. Final aggregation policy

Task-level mapping with the requested columns: `S19_aggregation_membership.tsv`.

Inventory: **11 core (8 speaker + 3 content), 3 diagnostic (all channel).**

| Aggregate | Tasks entering | MFCC baseline included? |
|---|---|---|
| Capability heatmap | all 14 | **yes** — it is a row, shaded within column |
| Plain mean | all 14 | **no** — excluded from all model rankings |
| Borda ranking | all 14 | **no** |
| Capability-balanced | **11** (speaker + content; channel excluded) | **no** |
| Sinhala speaker mean for MAEB correlations | 8 speaker tasks | **no** |
| Sinhala content mean for MAEB correlations | 3 content tasks | **no** |

**The four-core / ten-diagnostic policy no longer stands.** It demoted every task on
which pooled MFCCs matched the encoders, which removed seven genuine speaker-identity
tasks. The layer study then established that MFCC competitiveness is a property of
*final-layer extraction* — at a development-selected layer the encoders score about
twice the baseline — so that criterion was measuring the extraction default, not task
validity.

The policy now rests on **label provenance**, which no baseline affects. Three tasks are
demoted because their labels are not what the capability claims:

- `VoxLingua107SinhalaVideoClustering` — automatic video ids; one video may hold several speakers
- `WorldSpeechSinhalaSessionClustering` — sitting days, not speakers
- `WorldSpeechSinhalaQualityClassification` — DNSMOS terciles chosen in this work

**Consequence for a previously reported result.** With a four-task core, the core-only
ranking was anti-correlated with the full fourteen (τ = −0.137) and I concluded that
removing diagnostics reorders the table. With the eleven-task core, τ = **+0.687**,
while capability reweighting sits at τ = +0.152 against the original. **It is the
reweighting that reorders the table, not the removal.** See
`S3_aggregation_decomposed.tsv`.

---

## 5. Probe checks

Supplied: `S9_gpu_probe_vs_sklearn.tsv`, `S10_probe_convergence.tsv`.

### Stronger reference comparison — `S12_probe_reference_hard.tsv`

Your conditions, met: accuracy below ceiling, identical splits, train-only
standardisation, matched L2 objective (`C=1.0` in sklearn; `l2/(2n)·‖W‖²` with
`l2=1.0` on GPU — equivalent up to the bias term, which the GPU version does not
penalise and sklearn does not either).

| Representation | GPU acc | sklearn acc | GPU train CE | sklearn train CE | Same prediction |
|---|---|---|---|---|---|
| hubert, final layer | 0.4266 | 0.4458 | 0.0212 | 0.0211 | 92.5% |
| whisper, final layer | 0.7500 | 0.7500 | 0.0073 | 0.0074 | 94.9% |
| hubert, selected layer 3 | 0.9948 | 0.9930 | 0.0028 | 0.0026 | 99.8% |

Both solvers reach the same optimum — training cross-entropy agrees to **0.0003** — but
disagree on up to **7.5%** of individual predictions where the model is uncertain, with
accuracy differing by up to **0.019**. The correct description is *equivalent in
objective, not identical in predictions*.

### Convergence for all reported representations — `S15_convergence_all_conditions.tsv`

18 conditions: six encoders × {selected, final, all-layer mean}. Iterations, final loss,
gradient norm and cap-reached flag for each.

| Condition | Hit the 200-iteration cap |
|---|---|
| selected layer | **0 of 6** |
| final layer | **6 of 6** |
| all-layer mean | **5 of 6** |

The baselines were systematically truncated while the comparator was not, which would
inflate the reported layer gain.

### Inspection before rerunning — `S16_cap_sensitivity.tsv`

Refitting all 18 conditions at a 2,000-iteration budget: every one converges (145–460
iterations), and the largest accuracy change anywhere is **0.0007**. Mean change is
slightly **negative** for both baselines, so the cap marginally favoured them and the
reported gains of 0.3–0.8 are conservative rather than inflated.

**Conclusion: the problematic fits do not require rerunning the reported tables.**

---

## 6. Strict-duration robustness

Tables: `S7_matching_triplet_ids.tsv` (all 10,346 triplets with a per-query fallback
flag), `S11_strict_duration_subset.tsv` (the comparison).

- Retained count: **8,816 confirmed** — queries where *both* the content and speaker
  candidate satisfied the 1.5× ratio. 1,530 (14.8%) fell back.
- Retained speakers (bootstrap units): **476**, unchanged — the fallback does not remove
  any speaker entirely.

| Encoder | All 10,346 | Strict 8,816 | 95% CI (strict) | Δ |
|---|---|---|---|---|
| whisper-large-v3 | 0.6093 | **0.6933** | 0.679 – 0.708 | +0.0840 |
| hubert-large-ls960-ft | 0.5665 | 0.6233 | 0.610 – 0.637 | +0.0568 |
| wavlm-large | 0.4314 | 0.4800 | 0.464 – 0.496 | +0.0487 |
| xls-r-300m | 0.3949 | 0.4073 | 0.396 – 0.419 | +0.0124 |
| mms-300m | 0.2819 | 0.3106 | 0.299 – 0.323 | +0.0286 |
| w2v-bert-2.0 | 0.1755 | 0.1840 | 0.172 – 0.196 | +0.0085 |

**Ordering identical. Preference direction unchanged for every encoder — every interval
still excludes 0.5.** All six shift in the same direction, so the fallback queries were
depressing content preference; the strict subset is the more favourable reading.

No new extraction was needed.

---

## 7. Evidence files

All in this archive, at full precision.

| Requested | Supplied as |
|---|---|
| Complete Sinhala matrix | `results/S1_full_matrix_long.tsv` |
| Original per-task scores | `results/simteb_audio_results_per_task.tsv` |
| Layer-transfer, seed, classification, curve tables | `results/{J,O,Q,Q2,T,U,V}_*.tsv` |
| Held-out content-layer results | `results/J_layer_transfer.tsv`, rows with `capability=content` |
| Classification split IDs with post-filter counts | `results/S13_classification_split_ids.tsv` |
| Content split IDs | `results/S8_content_split_ids.tsv` |
| MAEB capability and per-model tables | `results/{N_maeb_comparison,N_maeb_per_model,S6_maeb_capability_matched}.tsv` |
| Original evaluator JSONs | `raw_results/sinhala/`, `raw_results/maeb/` |
| Experiment and task scripts | `code/`, `tasks/` |
| Launch configurations | `jobs/` |
| Environment and commit | `env/environment.txt`, `env/pip_freeze.txt` |
| Quality-tercile cut points | **recovered** — see below |

**DNSMOS tercile cut points: `low < 2.3101 ≤ medium < 2.5087 ≤ high`**, re-derived from
the cached source shards over all 20,981 scored segments (range 1.127–3.120, mean
2.401). They are computed at preparation time and not stored in the dataset, so this is
a re-derivation from the same source rather than a recovered log.

Classification split, after the ≥4-example filter:

| Half | Utterances | Speakers | After filter |
|---|---|---|---|
| Selection | 5,577 | 476 | 5,574 utts, 475 speakers |
| Scoring | 5,578 | 476 | 5,578 utts, 476 speakers |

---

## Not available / not yet run

- **Padding audit for four of six encoders** — not yet run. Covered: `wavlm-large`,
  `whisper-large-v3`. The other four share a masking code path with WavLM but were not
  themselves tested.
- **Padding audit for Whisper is not a padding test** — Whisper's feature extractor pads
  every input to 30 s regardless of batch size, so that arm tests batch-composition
  invariance only. Whisper's mask cannot be checked by this method.
- **MTEB's own probe convergence** — not available. scikit-learn convergence warnings
  were filtered from the run logs, so per-cell non-convergence in the benchmark's own
  classification probe is unrecorded.
- **Dataset revisions as-used** — not available. Preparation pinned no revision. The
  commits in the environment record are the Hub's *current* heads, resolved during
  review, not verified as those downloaded at preparation time.
- **Dependency manifest** — not available as a curated spec. `env/pip_freeze.txt` is the
  authoritative record.
- **`P_mfcc_vs_layer` setting mismatch** — not measured. MFCC rows and encoder rows in
  that table use different `n_init` and `batch_size`, as noted in item 2.
- **DNSMOS producing version** — not available. The `dnsmos_ovr` field ships with
  WorldSpeech and upstream does not record which DNSMOS release produced it.
