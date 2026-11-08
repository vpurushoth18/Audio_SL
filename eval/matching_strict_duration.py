#!/usr/bin/env python3
"""Recompute the preference statistic on strictly duration-matched queries only.

The headline diagnostic uses all 10,346 queries, but 1,530 of them (14.8%) had
no candidate inside the 1.5x duration ratio for at least one slot and fell back
to an unrestricted candidate. Describing all 10,346 as duration-matched is
therefore wrong.

This recomputes the same statistic on the 8,816 queries where BOTH slots had an
in-ratio candidate, reusing the cached embeddings and the exported triplet
table. If the preference ordering survives, the fallback is immaterial; if it
moves, the headline number needs the restriction stated.

Writes eval/analysis/S11_strict_duration_subset.tsv.
"""
import os
import numpy as np
import pandas as pd

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
RNG = np.random.default_rng(0)
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def cluster_bootstrap(values, groups, n=10000, alpha=0.05):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    out = np.empty(n)
    for b in range(n):
        pick = uniq[RNG.integers(0, len(uniq), len(uniq))]
        out[b] = values[np.concatenate([idx[g] for g in pick])].mean()
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def main():
    tri = pd.read_csv(os.path.join(ROOT, "eval/analysis/S7_matching_triplet_ids.tsv"),
                      sep="\t")
    strict = tri[tri.dur_fallbacks == 0].reset_index(drop=True)
    print(f"  all queries: {len(tri)}   strictly matched: {len(strict)} "
          f"({len(strict)/len(tri):.1%})\n")

    rows = []
    for tag in sorted(NAMES):
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        E = np.nan_to_num(z["emb"][-1].astype(np.float32))
        A = l2(E)

        def pref(t):
            sc = (A[t.q.values] * A[t.content.values]).sum(1)
            ss = (A[t.q.values] * A[t.speaker.values]).sum(1)
            w = (sc > ss).astype(float)
            w[np.isclose(sc, ss, rtol=0, atol=1e-6)] = 0.5
            return w

        w_all, w_str = pref(tri), pref(strict)
        lo, hi = cluster_bootstrap(w_str, strict.q_speaker.values)
        rows.append(dict(model=NAMES[tag], tag=tag,
                         n_all=len(tri), p_all=float(w_all.mean()),
                         n_strict=len(strict), p_strict=float(w_str.mean()),
                         ci_lo=lo, ci_hi=hi,
                         delta=float(w_str.mean() - w_all.mean())))
        print(f"  {NAMES[tag]:34s} all {w_all.mean():.4f}  strict {w_str.mean():.4f}  "
              f"[{lo:.3f}, {hi:.3f}]  delta {w_str.mean()-w_all.mean():+.4f}", flush=True)
        del z, E, A

    d = pd.DataFrame(rows).sort_values("p_strict", ascending=False)
    out = os.path.join(ROOT, "eval/analysis/S11_strict_duration_subset.tsv")
    d.to_csv(out, sep="\t", index=False)
    same = list(d.sort_values("p_all", ascending=False).model) == list(d.model)
    print(f"\n  preference ordering identical on the strict subset: {same}")
    print(f"  largest shift: {d.delta.abs().max():.4f}")
    print(f"  all strict intervals still exclude 0.5: "
          f"{bool(((d.ci_lo > 0.5) | (d.ci_hi < 0.5)).all())}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
