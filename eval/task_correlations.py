#!/usr/bin/env python3
"""Spearman correlations between individual tasks, not between aggregates.

For each task t, the vector s_t = [s_1t ... s_Mt] over the M shared models.
Correlating those vectors asks whether models that do well on one task do well
on another - a question the capability aggregates cannot answer, because they
average exactly the variation of interest.

The grouping that matters is same-corpus/different-capability against
different-corpus/same-capability: if task design drives the ranking, the first
should be LOWER than the second.

Two cautions are built into the output rather than left to the reader:

  * near-ceiling tasks compress rank variance, so a low correlation there can
    be a range artefact. The per-task spread is reported alongside.
  * the 12 checkpoints include size variants of the same family, which are not
    independent observations. Intervals come from a bootstrap over models and
    will be wide; they are not a significance test.

Writes eval/analysis/task_correlations.tsv, task_pairs_summary.tsv,
task_group_comparison.tsv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
A = os.path.join(ROOT, "eval/analysis")
RNG = np.random.default_rng(0)

CORPUS = {
    "OpenSLR52SpeakerClustering": "OpenSLR-52", "OpenSLR52SpeakerClassification": "OpenSLR-52",
    "OpenSLR52SpeakerPairClassification": "OpenSLR-52", "OpenSLR52AudioRetrieval": "OpenSLR-52",
    "OpenSLR52AudioReranking": "OpenSLR-52",
    "OpenSLR30SpeakerClustering": "OpenSLR-30", "OpenSLR30SpeakerClassification": "OpenSLR-30",
    "OpenSLR30SpeakerPairClassification": "OpenSLR-30",
    "OmnilingualSinhalaSpeakerClustering": "Omnilingual",
    "OmnilingualSinhalaSpeakerClassification": "Omnilingual",
    "OmnilingualSinhalaContentRetrieval": "Omnilingual",
    "WorldSpeechSinhalaSessionClustering": "WorldSpeech",
    "WorldSpeechSinhalaQualityClassification": "WorldSpeech",
    "VoxLingua107SinhalaVideoClustering": "VoxLingua107",
}
CAPABILITY = {t: ("content" if ("Retrieval" in t or "Reranking" in t)
                  else "channel" if t.startswith(("WorldSpeech", "VoxLingua"))
                  else "speaker") for t in CORPUS}
FAMILY = lambda m: ("whisper" if "whisper" in m else "xls-r" if "xls-r" in m
                    else "other")                                   # noqa: E731


def boot_rho(x, y, n=10000):
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        out.append(stats.spearmanr(x[i], y[i]).statistic)
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    per = pd.read_csv(os.path.join(ROOT, "eval/simteb_audio_results_per_task.tsv"),
                      sep="\t").set_index("model")
    per = per[~per.index.str.startswith("baseline/")]        # control, not a model
    tasks = [t for t in per.columns if t in CORPUS]
    M = len(per)
    print(f"  {M} shared models x {len(tasks)} tasks\n")

    # per-task spread, so a ceiling effect is visible rather than inferred
    spread = pd.DataFrame({
        "task": tasks,
        "corpus": [CORPUS[t] for t in tasks],
        "capability": [CAPABILITY[t] for t in tasks],
        "min": [per[t].min() for t in tasks],
        "max": [per[t].max() for t in tasks],
        "range": [per[t].max() - per[t].min() for t in tasks],
        "sd": [per[t].std() for t in tasks],
    }).sort_values("range")
    print("  tasks with the least spread (ceiling/floor risk):")
    print(spread.head(4)[["task", "min", "max", "range"]].to_string(index=False,
          float_format=lambda v: f"{v:.2f}"))

    # pairwise correlations
    rows = []
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            x, y = per[a].values, per[b].values
            rho = stats.spearmanr(x, y).statistic
            lo, hi = boot_rho(x, y)
            same_corpus = CORPUS[a] == CORPUS[b]
            same_cap = CAPABILITY[a] == CAPABILITY[b]
            rows.append(dict(
                task_a=a, task_b=b, corpus_a=CORPUS[a], corpus_b=CORPUS[b],
                capability_a=CAPABILITY[a], capability_b=CAPABILITY[b],
                same_corpus=same_corpus, same_capability=same_cap,
                group=("same corpus, same capability" if same_corpus and same_cap
                       else "same corpus, different capability" if same_corpus
                       else "different corpus, same capability" if same_cap
                       else "different corpus, different capability"),
                spearman=rho, ci_lo=lo, ci_hi=hi, n_models=M,
                range_a=per[a].max() - per[a].min(),
                range_b=per[b].max() - per[b].min()))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(A, "task_correlations.tsv"), sep="\t", index=False,
             float_format="%.17g")
    spread.to_csv(os.path.join(A, "task_spread.tsv"), sep="\t", index=False,
                  float_format="%.17g")

    # the comparison the question hinges on
    g = d.groupby("group").spearman.agg(["count", "median", "mean", "min", "max"])
    g.to_csv(os.path.join(A, "task_group_comparison.tsv"), sep="\t",
             float_format="%.17g")
    print("\n  correlation by pair type:")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))

    # same-corpus pairs, which is what he asked to see first
    sc = d[d.same_corpus].sort_values("spearman")
    sc.to_csv(os.path.join(A, "task_pairs_summary.tsv"), sep="\t", index=False,
              float_format="%.17g")
    print(f"\n  same-corpus pairs, lowest correlations ({len(sc)} pairs total):")
    print(sc.head(6)[["task_a", "task_b", "corpus_a", "capability_a",
                      "capability_b", "spearman", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # a direct test of the framing
    a = d[d.group == "same corpus, different capability"].spearman
    b = d[d.group == "different corpus, same capability"].spearman
    print(f"\n  same corpus / different capability : median {a.median():+.3f} (n={len(a)})")
    print(f"  different corpus / same capability : median {b.median():+.3f} (n={len(b)})")
    print(f"  difference: {a.median() - b.median():+.3f}")
    if len(a) and len(b):
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  Mann-Whitney U p = {u.pvalue:.3f}  (descriptive; the pairs are "
              f"not independent)")

    # family dependence: how much of the signal is size variants?
    fam = pd.Series({m: FAMILY(m) for m in per.index})
    print(f"\n  checkpoint families: {dict(fam.value_counts())}")
    print("  (size variants within a family are not independent observations)")
    print(f"\nwrote task_correlations.tsv, task_pairs_summary.tsv, "
          f"task_group_comparison.tsv, task_spread.tsv")


if __name__ == "__main__":
    main()
