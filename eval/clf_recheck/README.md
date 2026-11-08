# Corrected classification check

Supersedes the `classification_splits.tsv` and `classification_iteration_check.tsv`
sent in the previous addendum. Those were computed on the wrong partitions.

## The error

The original experiment creates the generator once and lets its state advance:

```python
# eval/classification_476.py  (code/classification_476.py here)
rng = np.random.default_rng(0)        # ONCE, before the loop
for tag in sorted(NAMES):
    half = rng.permutation(idx)       # advances -> a different split per model
```

My export reset it inside the loop:

```python
for tag in sorted(NAMES):
    half = np.random.default_rng(0).permutation(i52)   # same split for every model
```

Because `hubert` is first in `sorted(NAMES)` it consumed the first draw either way, so
that one model was correct and the other five were not. The iteration check in that
file ran on those wrong partitions, so its numbers did not describe the reported
experiment.

## Reproduction, verified

`classification_reproduction_check.tsv` compares mean accuracy at a 200-iteration cap
against `U_classification_476_speakers.tsv`, per model and condition.

**18 of 18 within tolerance. Largest absolute difference 0.00000.**

A guard also confirms no model now shares another's scoring half — 0 of 5, where the
previous version would have shown 5.

## Files

| File | Contents |
|---|---|
| `classification_splits.tsv` | 66,930 rows: model, utt_id, speaker_id, `selection`/`scoring`, `kept_after_min4_filter`, and `fold_seed0/1/2` = train / test / excluded |
| `classification_iteration_check.tsv` | 54 rows: 6 models x 3 conditions x 3 seeds, with `acc_cap200`, `acc_cap2000`, `acc_delta`, `iters_*`, `cap200_reached`, `converged_by_2000`, `final_loss_*`, `grad_norm_*` |
| `classification_reproduction_check.tsv` | the comparison above |
| `code/classification_splits_and_iters.py` | this check |
| `code/classification_476.py` | the original experiment, for the RNG placement |
| `logs/classification_recheck.log` | full run output |

Filtering: **11 utterances** are dropped across all six models by the >=4-example
filter, all in selection halves. Model order, embedding row order and the filter are
unchanged from the original.

## Iteration result on the correct partitions

| Condition | Reached the 200 cap | Mean change 200 -> 2000 | Worst |
|---|---|---|---|
| selected | 3 of 18 | -0.00002 | -0.00045 |
| all-layer mean | 14 of 18 | -0.00015 | -0.00224 |
| final | **18 of 18** | -0.00067 | -0.00403 |

All 54 fits converge by 2,000 iterations. **Largest accuracy change anywhere: 0.00403.**

The pattern from before holds on the correct splits: the baselines are the fits that
truncate, not the selected layer. The mean change is negative for every condition, so
the cap slightly favoured the baselines and the reported gains of 0.3-0.8 are
conservative rather than inflated.

**The reported classification table does not need updating.**

## Previous numbers, now superseded

The earlier addendum reported 34 of 54 reaching the cap and a largest change of
0.00314. On the correct partitions those are 35 of 54 and 0.00403. The conclusion is
unchanged; the figures are not.
