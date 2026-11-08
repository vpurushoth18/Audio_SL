# Addendum: three targeted checks

Additions to the earlier evidence package, not a replacement. Everything here is
newly computed or newly corrected; nothing already sent is repeated.

All tables at full precision (`%.17g` where floats are written).

---

## 1. Classification on the exact reported splits

**`classification_splits.tsv`** — 66,930 rows, the real partition for every model.

| Column | Meaning |
|---|---|
| `model`, `utt_id`, `speaker_id` | identity |
| `partition` | `selection` or `scoring` half of OpenSLR-52 |
| `kept_after_min4_filter` | false if dropped for having fewer than 4 examples of its speaker |
| `fold_seed0/1/2` | `train`, `test`, or `excluded` for that seed's stratified split |

11,155 utterances × 6 models. **3 utterances per model are dropped** by the
≥4-example filter, all in the selection half, costing one speaker there (475 of 476
retained; the scoring half keeps all 476).

The permutation into halves uses `np.random.default_rng(0)` over the embedding row
order, which is identical across encoders, so the halves are shared between models.
The probe folds are per-seed and per-model and are recorded individually.

**`classification_iteration_check.tsv`** — 54 rows: 6 models × 3 conditions × 3 seeds,
on the scoring partition the reported table uses.

Columns: `acc_cap200`, `acc_cap2000`, `acc_delta`, `iters_cap200`, `iters_cap2000`,
`cap200_reached`, `converged_by_2000`, `final_loss_*`, `grad_norm_*`.

Result:

- **34 of 54 fits reach the 200-iteration cap.** They are concentrated in the
  `final` and `all_layer_mean` conditions; `selected` rarely reaches it.
- **All 54 converge by 2,000 iterations.**
- **Largest accuracy change 200 → 2000: 0.00314.**

**The reported classification table does not need updating.** The gains it reports are
0.3–0.8; the largest effect of the iteration cap is 0.003, and the mean effect is
negative, meaning the cap marginally favoured the baselines rather than the selected
layer.

---

## 2. Matched MFCC–encoder clustering comparison

This replaces `P_mfcc_vs_layer.tsv`, which was **not a fair comparison**: its MFCC rows
used `n_init=5, batch_size=512` with 3 seeds while its encoder rows were carried over
from `layer_transfer.py` at `n_init=3, batch_size=1024` with 1 seed.

**`mfcc_matched_per_seed.tsv`** — AMI for every arm × seed.
**`mfcc_matched_comparison.tsv`** — means, standard deviations and recomputed ratios.

Identical across all three arms:

```
utterances    OpenSLR-30, 2,064 utterances, 12 speakers
labels        speaker_id from the same manifest
seeds         0, 1, 2, 3, 4
clusterer     MiniBatchKMeans(n_clusters=12, n_init=5, batch_size=1024,
                              max_iter=100, tol=0.0, max_no_improvement=10)
input         L2-normalised
layer choice  still made on development data (OpenSLR-52), frozen
```

| Arm | AMI mean ± sd |
|---|---|
| MFCC-40 | **0.4079 ± 0.0258** |
| hubert-large, final | 0.1916 ± 0.0036 |
| hubert-large, selected (L3) | 0.8565 ± 0.0151 |
| wavlm-large, final | 0.3335 ± 0.0218 |
| wavlm-large, selected (L3) | 0.8266 ± 0.0227 |
| whisper-large-v3, final | 0.2089 ± 0.0053 |
| whisper-large-v3, selected (L13) | 0.8306 ± 0.0360 |
| w2v-bert-2.0, final | 0.5382 ± 0.0150 |
| w2v-bert-2.0, selected (L17) | 0.8160 ± 0.0222 |
| mms-300m, final | 0.2121 ± 0.0084 |
| mms-300m, selected (L4) | 0.7847 ± 0.0089 |
| xls-r-300m, final | 0.0645 ± 0.0041 |
| xls-r-300m, selected (L4) | 0.7534 ± 0.0270 |

| Condition | Encoders above MFCC | Median ratio | Range |
|---|---|---|---|
| Final layer | **1 of 6** | **0.52×** | 0.16–1.32× |
| Selected layer | **6 of 6** | **2.01×** | 1.85–2.10× |

**The headline survives the fair protocol.** The unmatched table reported 0.55× and
2.00× with the same 1/6 and 6/6 counts; matched, it is 0.52× and 2.01×. The protocol
difference was real but contributed almost nothing to the effect.

---

## 3. Benchmark matrix rebuilt from raw JSON

**`benchmark_scores.tsv`** — 182 rows, 13 models × 14 tasks, read from the result JSONs
rather than from any intermediate table.

Columns: `task`, `model`, `split`, `hf_subset`, `score`, `score_key`, `metric`,
`value_at_declared_key`, `other_keys_equal_to_main`, `capability`,
`core_or_diagnostic`, `corpus`.

### Metric labels corrected

The metrics are unchanged; the labels were wrong in earlier exports.

| Family | Score key | Label | Previously mislabelled as |
|---|---|---|---|
| Clustering | `v_measure` | V-measure | — |
| Classification | `accuracy` | accuracy | — |
| Pair classification | `max_ap` | max average precision | — |
| Retrieval | `hit_rate_at_5` | hit rate@5 | nDCG@10 |
| Reranking | `map_at_5` | MAP@5 | nDCG@10 |

`score_key` comes from the **task declaration**, not from a numeric match. Several keys
can tie with `main_score` — `cosine_ap` and `manhattan_ap` often equal `max_ap` — so a
numeric match picks the wrong name. `other_keys_equal_to_main` records the ties.
Verification: **0 rows where `main_score` differs from the value stored under the
declared key.**

**`aggregate_ranks.tsv`** — every aggregate side by side, with mean rank per capability.
The MFCC baseline is excluded from all rankings.

| Comparison | Kendall τ |
|---|---|
| plain mean vs Borda | +0.545 |
| plain mean vs core-only (11 tasks) | +0.687 |
| plain mean vs capability-balanced | +0.606 |
| core-only vs capability-balanced | +0.412 |

**`transfer_correlations.tsv`** — 8 comparisons over 12 checkpoints, bootstrap over
checkpoints (n=10,000, α=0.05, seed 0).

| MAEB subset | → Sinhala | ρ | 95% CI | Excludes 0 |
|---|---|---|---|---|
| content | content | **0.979** | 0.870 – 1.000 | yes |
| all | content | 0.937 | 0.692 – 1.000 | yes |
| non-speech | content | 0.853 | 0.421 – 1.000 | yes |
| speaker | content | 0.685 | 0.157 – 0.957 | yes |
| speaker | speaker | 0.399 | −0.380 – 0.886 | no |
| non-speech | speaker | 0.259 | −0.491 – 0.794 | no |
| all | speaker | 0.161 | −0.561 – 0.730 | no |
| content | speaker | 0.133 | −0.570 – 0.737 | no |

**`evaluation_config.json`** — metrics by family, the label correction, the reranking
candidate structure, aggregation policy, bootstrap settings, precision.

---

## Reranking: implementation clarification, not a rerun

**The evaluation uses one global candidate corpus, not query-specific pools.** Your
reading of the code is correct.

`code/task_code/openslr_retrieval.py`, `OpenSLR52AudioReranking._build`:

```python
corpus  = src.select_columns(["audio"]).add_column("id", utt_ids)   # ALL 2,384 rows
qrels   = {query_id: {positive_id: 1} ...}
q_idx   = [i for i, u in enumerate(utt_ids) if u in qrels]
queries = src.select(q_idx)...                                      # the 500
```

Every one of the 500 queries is scored against all 2,384 candidates. The same-speaker
hard negatives exist in the dataset construction and sit in the corpus, but they do not
form a per-query boundary.

**Retaining this setup and describing it accurately.** Earlier wording of mine —
"reranking narrows the candidate pool" — is withdrawn. Switching to per-query pools
would require rerunning the task for all 13 models and would make it an easier
evaluation; the global corpus is the harder and more standard setting.

---

## Files

```
classification_splits.tsv              66,930 rows   partition + per-seed folds
classification_iteration_check.tsv         54 rows   200 vs 2000, per seed
mfcc_matched_comparison.tsv                12 rows   matched ratios
mfcc_matched_per_seed.tsv                  65 rows   AMI per arm per seed
benchmark_scores.tsv                      182 rows   rebuilt from raw JSON
aggregate_ranks.tsv                        12 rows   all aggregates
transfer_correlations.tsv                   8 rows   MAEB to Sinhala
evaluation_config.json                               settings behind the above
code/                                                the three scripts + task code
logs/                                                run logs
```

## Not run

- No new extraction, model, or corpus was used. Everything reuses cached embeddings
  and existing result JSONs.
- The reported classification table was **not** updated, because the iteration check
  showed a maximum effect of 0.003 against gains of 0.3–0.8.
- `P_mfcc_vs_layer.tsv` in the earlier package is **superseded** by
  `mfcc_matched_comparison.tsv` and should not be used.
