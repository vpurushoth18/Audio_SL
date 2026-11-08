#!/usr/bin/env python3
"""Analyses for the paper, with the statistics a reviewer will ask for.

Covers the questions raised in review of the benchmark:

  A. Ranking stability and aggregation        (how much does the scheme matter?)
  B. Task redundancy                          (are tasks measuring one thing?)
  C. Task saturation                          (do tasks separate the models?)
  D. Model scale                              (does bigger help, and where?)
  E. Pre-training language                    (English-only vs multilingual)
  F. Speaker vs content trade-off             (is there a real trade-off?)
  G. Cross-dataset robustness                 (does a result transfer corpora?)

Design decisions that matter for correctness, stated here because they bound
every number below:

1. WE HAVE 12 MODELS AND 14 TASKS. That is the sample size for everything. A
   Spearman correlation over 12 points has a 95% CI roughly +/-0.5 wide, so
   point estimates alone are not evidence. Every correlation here is reported
   with a bootstrap CI, and the script refuses to call anything significant on
   n=12 without one.

2. SCORES ARE NOT COMMENSURABLE ACROSS TASK TYPES. Accuracy, V-measure, max AP
   and hit-rate@5 are different scales with different chance levels; averaging
   them is not meaningful even though the leaderboard does it. Aggregation here
   is rank-based by default, with the mean reported only for comparability with
   the leaderboard convention.

3. BOOTSTRAP UNIT DEPENDS ON THE CLAIM. For "is this ranking stable", the
   resampled unit is the TASK (we are asking about generalisation to other
   tasks). For "do these two tasks measure the same thing", the unit is the
   MODEL. Mixing these up inflates confidence.

4. WE CANNOT TEST ENGLISH-ONLY vs MULTILINGUAL PROPERLY. There are 2 English-only
   models and 10 multilingual ones. The script reports the comparison with that
   imbalance stated, as a rank position rather than a significance test, because
   a two-sample test on n=2 is not informative.

Writes eval/analysis/*.tsv and prints a report.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
RNG = np.random.default_rng(0)
B = 10000           # bootstrap resamples

# ---------------------------------------------------------------- taxonomy
# Which tasks ask "who is speaking" vs "what is being said". This split drives
# analysis F, so it is declared explicitly rather than inferred from names.
SPEAKER_TASKS = [
    "OpenSLR52SpeakerClustering", "OpenSLR30SpeakerClustering",
    "OpenSLR52SpeakerClassification", "OpenSLR30SpeakerClassification",
    "OpenSLR52SpeakerPairClassification", "OpenSLR30SpeakerPairClassification",
    "OmnilingualSinhalaSpeakerClustering",
    "OmnilingualSinhalaSpeakerClassification",
]
CONTENT_TASKS = [
    "OpenSLR52AudioRetrieval", "OpenSLR52AudioReranking",
    "OmnilingualSinhalaContentRetrieval",
]
# Neither: these label the recording or the channel, not speaker or content.
CHANNEL_TASKS = [
    "WorldSpeechSinhalaSessionClustering", "VoxLingua107SinhalaVideoClustering",
    "WorldSpeechSinhalaQualityClassification",
]

# Capability per task, for the capability-balanced aggregate. Channel tasks are
# kept OUT of the aggregate: a VoxLingua video id or a WorldSpeech sitting day
# is not a speaker label, so they are diagnostics rather than a capability.
CAPABILITY = {
    **{t: "speaker" for t in SPEAKER_TASKS},
    **{t: "content" for t in CONTENT_TASKS},
    **{t: "channel" for t in CHANNEL_TASKS},
}
# Which corpus each task draws from, for leave-one-corpus-out.
CORPUS = {
    "OpenSLR52SpeakerClustering": "OpenSLR-52",
    "OpenSLR52SpeakerClassification": "OpenSLR-52",
    "OpenSLR52SpeakerPairClassification": "OpenSLR-52",
    "OpenSLR52AudioRetrieval": "OpenSLR-52",
    "OpenSLR52AudioReranking": "OpenSLR-52",
    "OpenSLR30SpeakerClustering": "OpenSLR-30",
    "OpenSLR30SpeakerClassification": "OpenSLR-30",
    "OpenSLR30SpeakerPairClassification": "OpenSLR-30",
    "OmnilingualSinhalaSpeakerClustering": "Omnilingual",
    "OmnilingualSinhalaSpeakerClassification": "Omnilingual",
    "OmnilingualSinhalaContentRetrieval": "Omnilingual",
    "WorldSpeechSinhalaSessionClustering": "WorldSpeech",
    "WorldSpeechSinhalaQualityClassification": "WorldSpeech",
    "VoxLingua107SinhalaVideoClustering": "VoxLingua107",
}

# Pre-training language exposure, from each model's own documentation.
ENGLISH_ONLY = ["microsoft/wavlm-large", "facebook/hubert-large-ls960-ft"]
# Model families with >= 3 sizes, for the scale analysis.
FAMILIES = {
    "Whisper": [
        ("openai/whisper-tiny", 0.039), ("openai/whisper-base", 0.074),
        ("openai/whisper-small", 0.244), ("openai/whisper-medium", 0.769),
        ("openai/whisper-large-v3", 1.550),
    ],
    "XLS-R": [
        ("facebook/wav2vec2-xls-r-300m", 0.300),
        ("facebook/wav2vec2-xls-r-1b", 1.000),
        ("facebook/wav2vec2-xls-r-2b", 2.000),
    ],
}


# ---------------------------------------------------------------- helpers
def boot_ci(fn, data, n=2000, alpha=0.05):
    """Percentile bootstrap CI for fn over rows of `data` (a list/array)."""
    data = np.asarray(data, dtype=object)
    stats_ = []
    k = len(data)
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        v = fn(data[idx])
        if v is not None and np.isfinite(v):
            stats_.append(v)
    if not stats_:
        return (np.nan, np.nan)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def boot_pair_ci(xa, xb, n=2000, alpha=0.05):
    """Percentile bootstrap CI for Spearman(xa, xb), resampling MODELS."""
    k = len(xa)
    if k < 4:
        return (np.nan, np.nan)
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, k, k)
        out[i] = spearman(xa[idx], xb[idx])
    out = out[np.isfinite(out)]
    if out.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def spearman(a, b):
    """Spearman without scipy's per-call overhead.

    Rank-transform then Pearson. Called ~1e6 times by the bootstraps, where
    scipy.spearmanr's validation dominates the runtime.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 3:
        return np.nan
    ra = stats.rankdata(a)
    rb = stats.rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return np.nan
    return float((ra * rb).sum() / denom)


def ranks(series):
    """1 = best. Higher score is better for every metric used here."""
    return series.rank(ascending=False, method="average")


def fmt_ci(lo, hi):
    return f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) else "[n/a]"


# ---------------------------------------------------------------- loading
def load(results_prefix, cache):
    long = pd.read_csv(f"{results_prefix}.tsv", sep="\t")
    wide = long.pivot_table(index="model", columns="task", values="main_score")
    types = long.drop_duplicates("task").set_index("task").task_type.to_dict()

    # per-cell variance where MTEB provides it: clustering gives v_measure_std
    # over its 10 bootstrap resamples, classification gives 10 experiments.
    var = {}
    for p in Path(cache).rglob("*.json"):
        if p.name.startswith("model_meta"):
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        if "task_name" not in blob or "scores" not in blob:
            continue
        parts = p.relative_to(cache).parts
        model = parts[parts.index("results") + 1].replace("__", "/") \
            if "results" in parts else None
        for entries in blob["scores"].values():
            for e in entries:
                sd = e.get("v_measure_std")
                if sd is None and "scores_per_experiment" in e:
                    vals = [x.get("accuracy") for x in e["scores_per_experiment"]
                            if isinstance(x, dict) and x.get("accuracy") is not None]
                    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
                if sd is not None and model:
                    var[(model, blob["task_name"])] = float(sd)
    return long, wide, types, var


# ------------------------------------------------- A. aggregation stability
def aggregation(wide, types, out, rep):
    rank_mat = wide.apply(ranks, axis=0)                 # task-wise ranks

    schemes = pd.DataFrame(index=wide.index)
    schemes["mean_score"] = wide.mean(axis=1)            # leaderboard convention
    schemes["mean_rank"] = -rank_mat.mean(axis=1)        # negate: higher=better
    # Borda, as the MTEB leaderboard computes it
    n_models = len(wide)
    schemes["borda"] = (n_models - rank_mat).sum(axis=1)
    # per-task-type mean of ranks, so a type with many tasks cannot dominate
    tt = pd.Series({t: types.get(t, "?") for t in wide.columns})
    schemes["mean_rank_by_type"] = -(
        rank_mat.T.groupby(tt).mean().mean(axis=0))

    order = {c: schemes[c].rank(ascending=False, method="min").astype(int)
             for c in schemes}
    order = pd.DataFrame(order)
    order.to_csv(os.path.join(out, "A_aggregation_ranks.tsv"), sep="\t")

    rep.append("## A. Ranking depends on the aggregation scheme\n")
    rep.append(order.sort_values("borda").to_string())

    rep.append("\nAgreement between schemes (Kendall tau):")
    for a, b in itertools.combinations(order.columns, 2):
        tau = stats.kendalltau(order[a], order[b]).statistic
        rep.append(f"  {a:20s} vs {b:20s} tau = {tau:+.3f}")

    # Stability: resample TASKS, because the question is whether the ranking
    # would survive a different (equally arbitrary) choice of tasks.
    tasks = list(wide.columns)
    top1, rank_draws = {}, {m: [] for m in wide.index}
    for _ in range(B):
        sel = [tasks[i] for i in RNG.integers(0, len(tasks), len(tasks))]
        rm = wide[sel].apply(ranks, axis=0)
        borda = (n_models - rm).sum(axis=1)
        r = borda.rank(ascending=False, method="min")
        for m in wide.index:
            rank_draws[m].append(r[m])
        top1[borda.idxmax()] = top1.get(borda.idxmax(), 0) + 1

    stab = pd.DataFrame({
        "borda_rank": order["borda"],
        "rank_ci_lo": {m: np.percentile(rank_draws[m], 2.5) for m in wide.index},
        "rank_ci_hi": {m: np.percentile(rank_draws[m], 97.5) for m in wide.index},
        "p_is_best": {m: top1.get(m, 0) / B for m in wide.index},
    }).sort_values("borda_rank")
    stab.to_csv(os.path.join(out, "A_rank_stability.tsv"), sep="\t")

    rep.append("\nRank stability, resampling the 14 tasks (B=%d):" % B)
    rep.append(stab.to_string())
    rep.append(
        "\n  Read this as the headline caveat: with 14 tasks the 95% rank\n"
        "  interval spans several positions for most models, so only large\n"
        "  rank gaps are interpretable. P(is best) is the honest statement of\n"
        "  which model leads.")
    return rank_mat, order


# ------------------------- A2. capability-balanced aggregate + LOCO
def capability_aggregate(wide, out, rep):
    """Average ranks within a capability, then weight capabilities equally.

    Plain means over tasks let whichever capability has the most tasks decide
    the winner: 8 of the 14 tasks here are speaker tasks. This removes that
    weighting, and keeps channel diagnostics out of the aggregate entirely.
    """
    rank_mat = wide.apply(ranks, axis=0)
    cap = pd.Series({t: CAPABILITY.get(t, "other") for t in wide.columns})

    per_cap = {}
    for c in sorted(set(cap.values)):
        cols = [t for t in wide.columns if cap[t] == c]
        if cols:
            per_cap[c] = rank_mat[cols].mean(axis=1)
    per_cap = pd.DataFrame(per_cap)

    scored = [c for c in per_cap.columns if c != "channel"]
    bal = -per_cap[scored].mean(axis=1)          # negate: higher is better
    out_df = per_cap.copy()
    out_df["capability_balanced_rank"] = bal.rank(ascending=False,
                                                  method="min").astype(int)
    out_df["plain_borda_rank"] = (
        (len(wide) - rank_mat).sum(axis=1).rank(ascending=False, method="min")
        .astype(int))
    out_df = out_df.sort_values("capability_balanced_rank")
    out_df.to_csv(os.path.join(out, "A2_capability_balanced.tsv"), sep="\t")

    rep.append("\n\n## A2. Capability-balanced aggregate\n")
    rep.append("Mean RANK within each capability (1 = best); channel tasks are\n"
               "reported but excluded from the aggregate.\n")
    rep.append(out_df.round(2).to_string())
    tau = stats.kendalltau(out_df.capability_balanced_rank,
                           out_df.plain_borda_rank).statistic
    rep.append(f"\n  Kendall tau vs plain Borda = {tau:+.3f}. "
               f"{sum(cap == 'speaker')} of {len(wide.columns)} tasks are "
               f"speaker tasks,\n  so a plain aggregate is largely a speaker "
               f"ranking.")

    # ---- leave-one-corpus-out
    corp = pd.Series({t: CORPUS.get(t, "?") for t in wide.columns})
    n_models = len(wide)
    full = (n_models - rank_mat).sum(axis=1).rank(ascending=False, method="min")
    rows = []
    for c in sorted(set(corp.values)):
        keep = [t for t in wide.columns if corp[t] != c]
        if len(keep) < 3:
            continue
        r = (n_models - wide[keep].apply(ranks, axis=0)).sum(axis=1) \
            .rank(ascending=False, method="min")
        rows.append(dict(dropped_corpus=c, n_tasks_dropped=len(wide.columns)-len(keep),
                         kendall_vs_full=stats.kendalltau(full, r).statistic,
                         new_best=r.idxmin(), max_rank_shift=int((full-r).abs().max())))
    loco = pd.DataFrame(rows).sort_values("kendall_vs_full")
    loco.to_csv(os.path.join(out, "A2_leave_one_corpus_out.tsv"), sep="\t",
                index=False)
    rep.append("\n\nLeave-one-corpus-out:")
    rep.append(loco.to_string(index=False))
    rep.append("\n  If new_best changes when a corpus is dropped, the headline\n"
               "  recommendation is corpus-dependent and should be a shortlist.")
    return out_df, loco


# ------------------------------------------------------- B. task redundancy
def redundancy(wide, types, out, rep):
    tasks = list(wide.columns)
    rows = []
    for a, b in itertools.combinations(tasks, 2):
        sub = wide[[a, b]].dropna()
        if len(sub) < 5:
            continue
        r = spearman(sub[a], sub[b])
        xa, xb = sub[a].values, sub[b].values
        lo, hi = boot_pair_ci(xa, xb)
        rows.append(dict(task_a=a, task_b=b, n_models=len(sub), spearman=r,
                         ci_lo=lo, ci_hi=hi,
                         same_type=types.get(a) == types.get(b)))
    red = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    red.to_csv(os.path.join(out, "B_task_correlations.tsv"), sep="\t", index=False)

    rep.append("\n\n## B. Task redundancy\n")
    rep.append("Most correlated task pairs (Spearman over 12 models, 95% CI):")
    for _, r in red.head(8).iterrows():
        rep.append(f"  {r.spearman:+.2f} {fmt_ci(r.ci_lo, r.ci_hi):>16s}  "
                   f"{r.task_a} ~ {r.task_b}")
    rep.append("\nLeast correlated:")
    for _, r in red.tail(5).iterrows():
        rep.append(f"  {r.spearman:+.2f} {fmt_ci(r.ci_lo, r.ci_hi):>16s}  "
                   f"{r.task_a} ~ {r.task_b}")

    n_wide = (red.ci_hi - red.ci_lo).median()
    rep.append(f"\n  Median CI width is {n_wide:.2f}. With 12 models these "
               f"correlations cannot\n  separate 0.6 from 0.95, so they support "
               f"'these tasks behave similarly',\n  not a precise redundancy "
               f"estimate.")

    # Leave-one-task-out: a more robust redundancy signal than correlation at
    # n=12. If dropping a task does not move the ranking, it is redundant.
    n_models = len(wide)
    full = (n_models - wide.apply(ranks, axis=0)).sum(axis=1).rank(ascending=False)
    loo = []
    for t in tasks:
        keep = [x for x in tasks if x != t]
        r = (n_models - wide[keep].apply(ranks, axis=0)).sum(axis=1) \
            .rank(ascending=False)
        loo.append(dict(task=t, task_type=types.get(t),
                        kendall_vs_full=stats.kendalltau(full, r).statistic,
                        max_rank_shift=int((full - r).abs().max())))
    loo = pd.DataFrame(loo).sort_values("kendall_vs_full")
    loo.to_csv(os.path.join(out, "B_leave_one_task_out.tsv"), sep="\t", index=False)
    rep.append("\nLeave-one-task-out influence on the overall ranking:")
    rep.append(loo.to_string(index=False))

    # Hierarchical clustering -> a proposed reduced benchmark
    mat = pd.DataFrame(1.0, index=tasks, columns=tasks)
    for _, r in red.iterrows():
        d = 1 - abs(r.spearman) if np.isfinite(r.spearman) else 1.0
        mat.loc[r.task_a, r.task_b] = mat.loc[r.task_b, r.task_a] = d
    arr = mat.to_numpy(dtype=float, copy=True)   # .values can be read-only
    np.fill_diagonal(arr, 0.0)
    arr = (arr + arr.T) / 2.0                    # enforce exact symmetry
    Z = hierarchy.linkage(squareform(arr, checks=False), method="average")
    for k in (4, 5, 6):
        lab = hierarchy.fcluster(Z, k, criterion="maxclust")
        groups = {}
        for t, c in zip(tasks, lab):
            groups.setdefault(c, []).append(t)
        rep.append(f"\n{k} task groups (average linkage on 1-|rho|):")
        for c, ts in sorted(groups.items()):
            rep.append(f"  group {c}: {', '.join(ts)}")
    return red


# ------------------------------------------------------- C. task saturation
def saturation(long, wide, types, var, out, rep):
    rows = []
    for t in wide.columns:
        v = wide[t].dropna()
        sds = [var.get((m, t)) for m in v.index]
        sds = [s for s in sds if s is not None]
        med_sd = float(np.median(sds)) if sds else np.nan
        spread = v.max() - v.min()
        rows.append(dict(
            task=t, task_type=types.get(t), n_models=len(v),
            min=v.min(), max=v.max(), spread=spread, mean=v.mean(),
            median_within_cell_sd=med_sd,
            # spread measured in units of within-cell noise: below ~2 means the
            # task cannot distinguish models beyond its own measurement error
            spread_over_noise=(spread / med_sd) if med_sd and med_sd > 0 else np.nan,
        ))
    sat = pd.DataFrame(rows).sort_values("spread")

    # Chance correction: for clustering tasks MTEB reports AMI alongside
    # V-measure. V-measure is not chance-corrected and inflates with many
    # clusters; AMI is. A large gap means the headline number is inflated.
    cl = long[long.task_type == "AudioClustering"].dropna(subset=["ami"])
    if not cl.empty:
        g = cl.groupby("task").agg(v_measure=("v_measure", "mean"),
                                   ami=("ami", "mean"),
                                   v_spread=("v_measure", lambda s: s.max()-s.min()),
                                   ami_spread=("ami", lambda s: s.max()-s.min()))
        g["inflation"] = g.v_measure - g.ami
        sat = sat.merge(g.reset_index(), on="task", how="left")
    sat.to_csv(os.path.join(out, "C_task_saturation.tsv"), sep="\t", index=False)

    rep.append("\n\n## C. Task saturation\n")
    rep.append(sat[["task", "task_type", "spread", "median_within_cell_sd",
                    "spread_over_noise"]].to_string(index=False))
    bad = sat[sat.spread_over_noise < 2]
    if not bad.empty:
        rep.append("\n  Tasks whose model spread is under 2x their own "
                   "measurement noise:")
        for _, r in bad.iterrows():
            rep.append(f"    {r.task} (spread {r.spread:.3f}, "
                       f"noise {r.median_within_cell_sd:.3f})")
        rep.append("  These cannot support claims about model differences.")
    if "inflation" in sat:
        inf = sat.dropna(subset=["inflation"]).sort_values(
            "inflation", ascending=False)
        rep.append("\nV-measure vs AMI on clustering tasks (chance correction):")
        rep.append(inf[["task", "v_measure", "ami", "inflation",
                        "v_spread", "ami_spread"]].to_string(index=False))
        rep.append("\n  AMI is chance-corrected, V-measure is not. A large gap\n"
                   "  means the V-measure headline is driven by cluster count\n"
                   "  rather than by the embedding.")
    return sat


# ------------------------------------------------------------- D. model scale
def scale(wide, types, out, rep):
    rep.append("\n\n## D. Model scale\n")
    rows = []
    tt = pd.Series({t: types.get(t, "?") for t in wide.columns})
    for fam, members in FAMILIES.items():
        members = [(m, p) for m, p in members if m in wide.index]
        if len(members) < 3:
            continue
        models = [m for m, _ in members]
        params = np.array([p for _, p in members])
        for ttype in sorted(set(tt.values)):
            cols = [c for c in wide.columns if tt[c] == ttype]
            if not cols:
                continue
            y = wide.loc[models, cols].mean(axis=1).values
            rows.append(dict(family=fam, task_type=ttype, n_sizes=len(models),
                             spearman_params_vs_score=spearman(params, y),
                             smallest=y[0], largest=y[-1], delta=y[-1] - y[0]))
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(out, "D_scale.tsv"), sep="\t", index=False)
    rep.append(sc.to_string(index=False))
    rep.append(
        "\n  Whisper has 5 sizes and XLS-R has 3. A rank correlation on 3-5\n"
        "  points cannot be significant, so these are reported as direction and\n"
        "  magnitude only. The interpretable claim is whether the SIGN is\n"
        "  consistent across task types within a family, not the coefficient.")
    return sc


# ------------------------------------------- E. pre-training language exposure
def pretraining(wide, types, out, rep):
    rep.append("\n\n## E. Pre-training language\n")
    eng = [m for m in ENGLISH_ONLY if m in wide.index]
    multi = [m for m in wide.index if m not in eng]
    rep.append(f"English-only: {len(eng)} models {eng}")
    rep.append(f"Multilingual: {len(multi)} models")
    rep.append("  The groups are 2 vs 10. No two-sample test on n=2 is\n"
               "  informative, so this is reported as where the English-only\n"
               "  models RANK inside each task type.")

    tt = pd.Series({t: types.get(t, "?") for t in wide.columns})
    rows = []
    for ttype in sorted(set(tt.values)):
        cols = [c for c in wide.columns if tt[c] == ttype]
        means = wide[cols].mean(axis=1)
        r = means.rank(ascending=False, method="min")
        for m in eng:
            rows.append(dict(task_type=ttype, model=m, rank=int(r[m]),
                             of_models=len(means), score=means[m],
                             best_multilingual=means[multi].max(),
                             gap_to_best_multi=means[m] - means[multi].max()))
    pt = pd.DataFrame(rows)
    pt.to_csv(os.path.join(out, "E_pretraining_language.tsv"), sep="\t", index=False)
    rep.append("\n" + pt.to_string(index=False))

    # Also split by speaker vs content, which is the substantive question
    for name, group in (("speaker", SPEAKER_TASKS), ("content", CONTENT_TASKS)):
        cols = [c for c in group if c in wide.columns]
        if not cols:
            continue
        means = wide[cols].mean(axis=1)
        r = means.rank(ascending=False, method="min")
        rep.append(f"\n{name} tasks: English-only ranks "
                   f"{[f'{m.split('/')[-1]}={int(r[m])}' for m in eng]} "
                   f"of {len(means)}")
    return pt


# --------------------------------------------- F. speaker vs content trade-off
def tradeoff(wide, out, rep):
    rep.append("\n\n## F. Speaker vs content trade-off\n")
    sp = [c for c in SPEAKER_TASKS if c in wide.columns]
    ct = [c for c in CONTENT_TASKS if c in wide.columns]
    if not sp or not ct:
        rep.append("  not enough tasks in one of the groups")
        return None

    # Rank within each group, so the two incommensurable metric families are
    # compared as orderings rather than as raw scores.
    sp_rank = wide[sp].apply(ranks, axis=0).mean(axis=1)
    ct_rank = wide[ct].apply(ranks, axis=0).mean(axis=1)
    df = pd.DataFrame({"speaker_mean_rank": sp_rank,
                       "content_mean_rank": ct_rank}).sort_values(
        "speaker_mean_rank")

    r = spearman(df.speaker_mean_rank, df.content_mean_rank)
    lo, hi = boot_pair_ci(df.speaker_mean_rank.values,
                          df.content_mean_rank.values, n=B)
    # permutation test: is the association stronger than chance in either
    # direction? (two-sided)
    obs = abs(r)
    perm = []
    a = df.speaker_mean_rank.values
    b = df.content_mean_rank.values
    for _ in range(B):
        perm.append(abs(spearman(a, RNG.permutation(b))))
    p = (1 + sum(x >= obs for x in perm)) / (B + 1)

    df.to_csv(os.path.join(out, "F_speaker_vs_content.tsv"), sep="\t")
    rep.append(df.to_string())
    rep.append(f"\nSpearman(speaker rank, content rank) = {r:+.3f} "
               f"{fmt_ci(lo, hi)}, permutation p = {p:.4f}")
    if r > 0.3 and lo > 0:
        rep.append("  POSITIVE rank correlation. Note both are ranks where 1 is\n"
                   "  best, so a positive correlation means models good at\n"
                   "  speaker tasks are ALSO good at content tasks: no trade-off.")
    elif r < -0.3 and hi < 0:
        rep.append("  NEGATIVE rank correlation: models good at speaker tasks are\n"
                   "  worse at content tasks. This is the trade-off claim, and the\n"
                   "  CI excludes zero.")
    else:
        rep.append("  The CI includes zero. With 12 models this analysis cannot\n"
                   "  establish a trade-off either way; report it as inconclusive\n"
                   "  and lean on the per-family pattern instead.")
    return df


# ------------------------------------------------- G. cross-dataset robustness
def cross_dataset(wide, out, rep):
    rep.append("\n\n## G. Cross-dataset robustness\n")
    pairs = [
        ("OpenSLR52SpeakerClustering", "OpenSLR30SpeakerClustering"),
        ("OpenSLR52SpeakerClustering", "OmnilingualSinhalaSpeakerClustering"),
        ("OpenSLR30SpeakerClustering", "OmnilingualSinhalaSpeakerClustering"),
        ("OpenSLR52SpeakerClassification", "OpenSLR30SpeakerClassification"),
        ("OpenSLR52SpeakerClassification",
         "OmnilingualSinhalaSpeakerClassification"),
        ("OpenSLR52SpeakerPairClassification",
         "OpenSLR30SpeakerPairClassification"),
        ("OpenSLR52AudioRetrieval", "OmnilingualSinhalaContentRetrieval"),
    ]
    rows = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        sub = wide[[a, b]].dropna()
        r = spearman(sub[a], sub[b])
        lo, hi = boot_pair_ci(sub[a].values, sub[b].values)
        rows.append(dict(task_a=a, task_b=b, n_models=len(sub), spearman=r,
                         ci_lo=lo, ci_hi=hi))
    cd = pd.DataFrame(rows)
    cd.to_csv(os.path.join(out, "G_cross_dataset.tsv"), sep="\t", index=False)
    rep.append(cd.to_string(index=False))
    rep.append("\n  Same task family, different corpus. High correlation means a\n"
               "  model's standing transfers; low means the corpus matters more\n"
               "  than the model, which would undermine single-corpus claims.")
    return cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "eval/simteb_audio_results"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "simteb_audio/mteb_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    long, wide, types, var = load(a.results, a.cache)

    # The MFCC baseline is a control, not a candidate model: it belongs in the
    # task-validity comparison, not in rankings, scaling or exposure analyses.
    # It is reported separately in the audit.
    baselines = [m for m in wide.index if m.startswith("baseline/")]
    if baselines:
        base = wide.loc[baselines]
        wide = wide.drop(index=baselines)
        long = long[~long.model.isin(baselines)]
        base.to_csv(os.path.join(a.out, "K_baseline_rows.tsv"), sep="\t")
    rep = [f"# Analysis: {len(wide)} models x {len(wide.columns)} tasks "
           f"({long.shape[0]} results)\n",
           f"Within-cell standard deviations available for {len(var)} cells "
           f"(clustering and classification only).\n"]
    if baselines:
        rep.append(f"Excluded from model comparisons and held out as controls: "
                   f"{', '.join(baselines)}. See K_baseline_rows.tsv and the "
                   f"audit for the task-validity comparison.\n")

    aggregation(wide, types, a.out, rep)
    capability_aggregate(wide, a.out, rep)
    redundancy(wide, types, a.out, rep)
    saturation(long, wide, types, var, a.out, rep)
    scale(wide, types, a.out, rep)
    pretraining(wide, types, a.out, rep)
    tradeoff(wide, a.out, rep)
    cross_dataset(wide, a.out, rep)

    text = "\n".join(str(x) for x in rep)
    Path(os.path.join(a.out, "report.md")).write_text(text)
    print(text)
    print(f"\n\nwrote {a.out}/report.md and the per-analysis TSVs")


if __name__ == "__main__":
    main()
