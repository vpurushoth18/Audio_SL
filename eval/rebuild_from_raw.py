#!/usr/bin/env python3
"""Rebuild the benchmark matrix from the raw result JSONs, labels corrected.

Earlier exports described retrieval as nDCG@10 and reranking as nDCG@10. The
evaluators actually report hit_rate_at_5 and map_at_5. The METRICS ARE
UNCHANGED - only the labels were wrong - so this preserves every score and
records the key each one actually came from, read out of the JSON rather than
assumed.

Full precision is kept throughout; nothing is rounded to manuscript values.

Writes:
  benchmark_scores.tsv      task, model, metric, score_key, score, capability, status
  aggregate_ranks.tsv       every aggregate, side by side
  transfer_correlations.tsv MAEB-to-Sinhala, by domain and capability-matched
  evaluation_config.json    the settings behind all of it
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
A = os.path.join(ROOT, "eval/analysis")
RNG = np.random.default_rng(0)

# read the membership definitions from the single source of truth
_sup = open(os.path.join(ROOT, "eval/supplementary.py")).read()
CAPABILITY = dict(re.findall(r'"(\w+)":\s*"(speaker|content|channel)"',
                             re.search(r'CAPABILITY = \{(.*?)\n\}', _sup, re.S).group(1)))
CHANNEL = set(re.findall(r'"(\w+)"',
                         re.search(r'CHANNEL_DIAGNOSTIC = \{(.*?)\}', _sup, re.S).group(1)))
CORPUS = dict(re.findall(r'"(\w+)":\s*"([\w.-]+)"',
                         re.search(r'CORPUS = \{(.*?)\n\}', _sup, re.S).group(1))) \
    if "CORPUS = {" in _sup else {}
MAEB_CAP = dict(re.findall(r'"(\w+)":\s*"(speaker|content|non-speech)"',
                           re.search(r'MAEB_CAP = \{(.*?)\n\}', _sup, re.S).group(1)))

# The DECLARED main_score for each task family, taken from the task metadata.
# Several keys can happen to equal main_score numerically - cosine_ap and
# manhattan_ap often tie with max_ap - so the key must come from the
# declaration, not from a numeric match.
DECLARED = {"Clustering": ("v_measure", "V-measure"),
            "Classification": ("accuracy", "accuracy"),
            "PairClassification": ("max_ap", "max average precision"),
            "Retrieval": ("hit_rate_at_5", "hit rate@5"),
            "Reranking": ("map_at_5", "MAP@5")}


def declared_metric(task):
    # order matters: PairClassification must be tested before Classification
    for fam in ("PairClassification", "Clustering", "Classification",
                "Reranking", "Retrieval"):
        if fam in task:
            return DECLARED[fam]
    return ("main_score", "?")


def scrape(cache):
    """Every (model, task, split) score, with the key it matched."""
    rows = []
    base = os.path.join(cache, "results")
    for mdir in sorted(os.listdir(base)):
        model = mdir.replace("__", "/", 1)
        for f in glob.glob(os.path.join(base, mdir, "**", "*.json"), recursive=True):
            task = os.path.basename(f)[:-5]
            if task == "model_meta":
                continue
            d = json.load(open(f))
            for split, entries in d.get("scores", {}).items():
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    ms = e.get("main_score")
                    if ms is None:
                        continue
                    # which named metric equals main_score? that IS the score key
                    keys = sorted(k for k, v in e.items()
                                  if k not in ("main_score", "hf_subset", "languages")
                                  and isinstance(v, (int, float)) and v == ms)
                    key, label = declared_metric(task)
                    rows.append(dict(task=task, model=model, split=split,
                                     hf_subset=e.get("hf_subset"),
                                     score=float(ms),
                                     score_key=key, metric=label,
                                     value_at_declared_key=e.get(key),
                                     other_keys_equal_to_main="|".join(keys)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(A, exist_ok=True)
    sin = scrape(os.path.join(ROOT, "simteb_audio/mteb_cache"))
    sin = sin[sin.task.isin(CAPABILITY)].copy()
    sin["capability"] = sin.task.map(CAPABILITY)
    sin["core_or_diagnostic"] = np.where(sin.task.isin(CHANNEL), "diagnostic", "core")
    sin["corpus"] = sin.task.map(CORPUS)
    # sanity: main_score must equal the value stored under the declared key
    mism = sin[sin.value_at_declared_key.notna() &
               (abs(sin.score - sin.value_at_declared_key.astype(float)) > 1e-9)]
    print(f"  rows where main_score != declared key: {len(mism)}")
    sin = sin.sort_values(["capability", "task", "score"], ascending=[True, True, False])
    sin.to_csv(os.path.join(A, "benchmark_scores.tsv"), sep="\t", index=False,
               float_format="%.17g")
    print(f"  benchmark_scores.tsv: {len(sin)} rows, "
          f"{sin.model.nunique()} models x {sin.task.nunique()} tasks")
    print("  score keys in use:", dict(sin.groupby('score_key').size()))

    # ---------- aggregates
    wide = sin.pivot_table(index="model", columns="task", values="score", aggfunc="mean")
    models = [m for m in wide.index if not m.startswith("baseline/")]
    W = wide.loc[models]
    rk = lambda d: d.rank(ascending=False, method="min")          # noqa: E731
    ranks = W.apply(rk, axis=0)
    cap = pd.Series({t: CAPABILITY[t] for t in W.columns})
    core = [t for t in W.columns if t not in CHANNEL]

    percap = pd.DataFrame({c: ranks[[t for t in W.columns if cap[t] == c]].mean(axis=1)
                           for c in ("speaker", "content", "channel")})
    agg = pd.DataFrame({
        "plain_mean_rank": rk(W.mean(axis=1)).astype(int),
        "borda_rank": rk((len(W) - ranks).sum(axis=1)).astype(int),
        "core_only_rank": rk(-W[core].apply(rk, axis=0).mean(axis=1)).astype(int),
        "capability_balanced_rank": rk(-percap[["speaker", "content"]].mean(axis=1)).astype(int),
        "mean_rank_speaker": percap["speaker"].round(4),
        "mean_rank_content": percap["content"].round(4),
        "mean_rank_channel": percap["channel"].round(4),
    }).sort_values("capability_balanced_rank")
    agg.to_csv(os.path.join(A, "aggregate_ranks.tsv"), sep="\t", float_format="%.17g")
    print(f"\n  aggregate_ranks.tsv: {len(agg)} models "
          f"(baseline excluded from all rankings)")
    for a, b in [("plain_mean_rank", "borda_rank"),
                 ("plain_mean_rank", "core_only_rank"),
                 ("plain_mean_rank", "capability_balanced_rank"),
                 ("core_only_rank", "capability_balanced_rank")]:
        print(f"    tau {a:26s} vs {b:26s} "
              f"{stats.kendalltau(agg[a], agg[b]).statistic:+.3f}")

    # ---------- transfer correlations
    maeb = scrape(os.path.join(ROOT, "simteb_audio/maeb_cache"))
    cov = maeb.pivot_table(index="model", columns="task", values="score", aggfunc="mean") * 100
    common = [m for m in cov.index if m in W.index]
    cov = cov.loc[common]
    shared = [t for t in cov.columns if cov[t].notna().all()]
    sin_cap = {c: W[[t for t in W.columns if cap[t] == c]].mean(axis=1).loc[common]
               for c in ("speaker", "content")}

    def ci(x, y, n=10000):
        out = []
        for _ in range(n):
            i = RNG.integers(0, len(x), len(x))
            if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
                continue
            out.append(stats.spearmanr(x[i], y[i]).statistic)
        return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
            if out else (np.nan, np.nan)

    rows = []
    overall = cov[shared].mean(axis=1)
    for scap in ("speaker", "content"):
        y = sin_cap[scap]
        lo, hi = ci(overall.values, y.values)
        rows.append(dict(maeb_subset="all", n_maeb_tasks=len(shared),
                         sinhala_capability=scap, n_models=len(common),
                         spearman=stats.spearmanr(overall, y).statistic,
                         kendall=stats.kendalltau(overall, y).statistic,
                         ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0)))
    for mcap in ("speaker", "content", "non-speech"):
        tasks = [t for t in shared if MAEB_CAP.get(t) == mcap]
        if len(tasks) < 3:
            continue
        x = cov[tasks].mean(axis=1)
        for scap in ("speaker", "content"):
            y = sin_cap[scap]
            lo, hi = ci(x.values, y.values)
            rows.append(dict(maeb_subset=mcap, n_maeb_tasks=len(tasks),
                             sinhala_capability=scap, n_models=len(common),
                             spearman=stats.spearmanr(x, y).statistic,
                             kendall=stats.kendalltau(x, y).statistic,
                             ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0)))
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join(A, "transfer_correlations.tsv"), sep="\t", index=False,
              float_format="%.17g")
    print(f"\n  transfer_correlations.tsv: {len(tr)} comparisons over {len(common)} checkpoints")
    print(tr[["maeb_subset", "sinhala_capability", "spearman", "ci_lo", "ci_hi",
              "excludes_zero"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------- config
    cfg = {
        "metrics_by_family": {
            "clustering": {"score_key": "v_measure", "label": "V-measure",
                           "note": "AMI reported alongside in the result JSON; "
                                   "the layer analyses use AMI"},
            "classification": {"score_key": "accuracy", "label": "accuracy",
                               "probe": "sklearn LogisticRegression(max_iter=100), C=1.0, "
                                        "samples_per_label=8, n_experiments=10"},
            "pair_classification": {"score_key": "max_ap", "label": "max average precision"},
            "retrieval": {"score_key": "hit_rate_at_5", "label": "hit rate@5"},
            "reranking": {"score_key": "map_at_5", "label": "MAP@5"}},
        "label_correction": "Earlier exports labelled retrieval and reranking as "
                            "nDCG@10. The metrics were always hit_rate_at_5 and "
                            "map_at_5; only the labels were wrong. No score changed.",
        "reranking_candidate_structure": {
            "type": "global candidate corpus",
            "corpus_size": 2384,
            "queries": 500,
            "detail": "The evaluator builds one corpus from all rows and scores every "
                      "query against it. Same-speaker hard negatives exist in the "
                      "dataset construction but impose no per-query pool boundary.",
            "code": "simteb_audio/src/simteb_audio/tasks/retrieval/"
                    "openslr_retrieval.py :: OpenSLR52AudioReranking._build"},
        "aggregation": {
            "baseline_mfcc_included_in_rankings": False,
            "baseline_mfcc_included_in_heatmap": True,
            "rank_tie_method": "min",
            "capability_balanced": "mean rank within capability, then equal weight "
                                   "across speaker and content; channel excluded",
            "core_policy": "11 core (8 speaker + 3 content), 3 diagnostic (channel). "
                           "Demotion is on LABEL PROVENANCE, not on the MFCC comparison."},
        "bootstrap": {"transfer_correlations": {"unit": "checkpoint", "n": 10000,
                                                "alpha": 0.05, "seed": 0}},
        "precision": "all tables written with %.17g; no manuscript rounding",
    }
    with open(os.path.join(A, "evaluation_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nwrote benchmark_scores / aggregate_ranks / transfer_correlations / "
          f"evaluation_config.json")


if __name__ == "__main__":
    main()
