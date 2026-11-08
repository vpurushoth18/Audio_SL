#!/usr/bin/env python3
"""Capability rankings, objective groups, MAEB-guided selection, rank correlation.

These numbers were previously computed ad hoc. This script is the recorded
form of them so they can be regenerated from the 45-checkpoint matrix.

Nothing here is a new experiment: every quantity is a transformation of
expansion_scores.tsv (SinAEB) and the MAEB result cache, both already written.

Outputs:
  Y1_capability_rankings.tsv        per-model speaker/content means + ranks
  Y2_objective_group_summary.tsv    the same aggregated by pretraining objective
  Y3_maeb_guided_selection.tsv      pick top-k on a MAEB subset, land where on SinAEB
  Y4_rank_correlation.tsv           speaker-vs-content ranking agreement
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expansion_analysis as EA

A = EA.A
TOPK = (1, 3, 5)


def main():
    W = pd.read_csv(os.path.join(A, "expansion_scores.tsv"), sep="\t", index_col=0)
    inv = pd.read_csv(os.path.join(A, "expansion_inventory.tsv"), sep="\t")
    meta = inv.set_index("model")

    sin_tasks = {c: [t for t in W.columns if EA.CAPABILITY.get(t) == c]
                 for c in ("speaker", "content")}
    cap = {c: W[ts].mean(axis=1) for c, ts in sin_tasks.items()}
    print(f"  {len(W)} checkpoints | SinAEB speaker tasks {len(sin_tasks['speaker'])}"
          f" content tasks {len(sin_tasks['content'])}")

    # ---------------- Y1 rankings
    d = pd.DataFrame({"speaker": cap["speaker"], "content": cap["content"]})
    for col in ("family", "objective", "encoder_params_m", "wave"):
        d[col] = [meta.loc[m, col] if m in meta.index else None for m in d.index]
    # 'min' so tied scores share the better rank and the next rank is skipped
    d["rank_speaker"] = d.speaker.rank(ascending=False, method="min").astype(int)
    d["rank_content"] = d.content.rank(ascending=False, method="min").astype(int)
    d["rank_gap"] = d.rank_speaker - d.rank_content
    d = d.sort_values("rank_speaker")
    d.index.name = "model"
    d.to_csv(os.path.join(A, "Y1_capability_rankings.tsv"), sep="\t",
             float_format="%.17g")
    print("\n  top 5 speaker:")
    print(d.nsmallest(5, "rank_speaker")[["speaker", "content", "rank_content",
                                          "objective", "encoder_params_m"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  top 5 content:")
    print(d.nsmallest(5, "rank_content")[["content", "speaker", "rank_speaker",
                                          "objective", "encoder_params_m"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))

    # ---------------- Y2 objective groups
    g = (d.groupby("objective")
           .agg(n=("speaker", "size"),
                speaker_mean=("speaker", "mean"), speaker_sd=("speaker", "std"),
                content_mean=("content", "mean"), content_sd=("content", "std"),
                params_median=("encoder_params_m", "median"))
           .sort_values("speaker_mean", ascending=False))
    g.to_csv(os.path.join(A, "Y2_objective_group_summary.tsv"), sep="\t",
             float_format="%.17g")
    print("\n  by objective:")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    assert int(g.n.sum()) == len(W), "objective counts must total the checkpoint count"

    # ---------------- Y3 MAEB-guided selection
    maeb = EA.scrape(os.path.join(EA.ROOT, "simteb_audio/maeb_cache"))
    C = (maeb[maeb.model.isin(W.index)]
         .pivot_table(index="model", columns="task", values="score") * 100
         ).reindex(W.index)
    shared = [t for t in C.columns if C[t].notna().all()]
    subsets = [("all", shared)] + [
        (c, [t for t in shared if EA.MAEB_CAP.get(t) == c])
        for c in ("speaker", "content", "non-speech")]

    rk = {c: cap[c].rank(ascending=False, method="min") for c in ("speaker", "content")}
    rows = []
    for name, ts in subsets:
        if len(ts) < 3:
            continue
        x = C[ts].mean(axis=1)
        for k in TOPK:
            pick = x.nlargest(k).index.tolist()
            rows.append(dict(
                maeb_subset=name, n_maeb_tasks=len(ts), top_k=k,
                picked="; ".join(pick),
                mean_sinaeb_speaker_rank=float(rk["speaker"][pick].mean()),
                mean_sinaeb_content_rank=float(rk["content"][pick].mean()),
                random_baseline_rank=(len(W) + 1) / 2, n_checkpoints=len(W)))
    sel = pd.DataFrame(rows)
    sel.to_csv(os.path.join(A, "Y3_maeb_guided_selection.tsv"), sep="\t",
               index=False, float_format="%.17g")
    print(f"\n  MAEB-guided selection (rank of {len(W)}, "
          f"random baseline {(len(W)+1)/2:.1f}):")
    print(sel[sel.top_k == 3][["maeb_subset", "mean_sinaeb_speaker_rank",
                               "mean_sinaeb_content_rank"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # ---------------- Y4 rank correlation
    rows = []
    for nm, (a, b) in {"speaker_vs_content": (cap["speaker"], cap["content"])}.items():
        sr = stats.spearmanr(a, b)
        kt = stats.kendalltau(a, b)
        # family-blocked CI on the same statistic, same scheme as X3
        rng = np.random.default_rng(0)
        fam = d.loc[a.index, "family"].values
        fams = np.unique(fam)
        by = {f: np.flatnonzero(fam == f) for f in fams}
        bs = []
        for _ in range(2000):
            idx = np.concatenate([by[f] for f in rng.choice(fams, len(fams), True)])
            bs.append(stats.spearmanr(a.values[idx], b.values[idx]).statistic)
        bs = np.array([v for v in bs if np.isfinite(v)])
        rows.append(dict(comparison=nm, n_checkpoints=len(W),
                         spearman=sr.statistic, spearman_p=sr.pvalue,
                         kendall=kt.statistic, kendall_p=kt.pvalue,
                         fam_ci_lo=float(np.percentile(bs, 2.5)),
                         fam_ci_hi=float(np.percentile(bs, 97.5)),
                         n_families=len(fams), n_boot=2000, seed=0))
    rc = pd.DataFrame(rows)
    rc.to_csv(os.path.join(A, "Y4_rank_correlation.tsv"), sep="\t", index=False,
              float_format="%.17g")
    print("\n  speaker-vs-content ranking agreement:")
    print(rc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  wrote Y1_capability_rankings, Y2_objective_group_summary, "
          "Y3_maeb_guided_selection, Y4_rank_correlation")


if __name__ == "__main__":
    main()
