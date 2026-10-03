#!/usr/bin/env python3
"""Does a general audio benchmark ranking predict the Sinhala ranking?

MAEB covers music, environmental sound, speech and audio-text tasks. Its
overall ranking therefore cannot, on its own, say anything about Sinhala
transfer: a model could place well on MAEB through music tasks alone. So the
comparison is run twice, against MAEB overall and against its speech subset
only, and the speech subset is the one that bears on the question.

Every checkpoint here was evaluated locally on both benchmarks with the same
extraction protocol, rather than read off a published leaderboard, so protocol
differences cannot explain a disagreement.

Reported for each comparison: Spearman, Kendall tau, the number of common
models, and a bootstrap interval over models. With a dozen checkpoints the
interval is wide, and that width is the result.

Writes eval/analysis/N_maeb_comparison.tsv.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
RNG = np.random.default_rng(0)

# MAEB(beta, audio-only) tasks grouped by domain. Only SPEECH bears on a claim
# about Sinhala speech representations.
DOMAIN = {
    "VoxPopuliGenderClustering": "speech",
    "VoxPopuliLanguageID": "speech",
    "VoxCelebSA": "speech",
    "CREMA_D": "speech",
    "CREMA_DClustering": "speech",
    "CommonLanguageAgeDetection": "speech",
    "MInDS14": "speech",
    "NMSQAPairClassification": "speech",
    "CREMADPairClassification": "speech",
    "IEMOCAPGender": "speech",
    "SIBFLEURS": "speech",
    "VoxPopuliAccentPairClassification": "speech",
    "BeijingOpera": "music",
    "MridinghamTonic": "music",
    "GTZANAudioReranking": "music",
    "GTZANGenre": "music",
    "JamAltArtistA2ARetrieval": "music",
    "BirdCLEF": "environmental",
    "VehicleSoundClustering": "environmental",
}


def load_maeb(cache):
    """Scrape main-score per (model, task) out of the MAEB result cache."""
    rows = []
    base = os.path.join(cache, "results")
    if not os.path.isdir(base):
        return pd.DataFrame(columns=["model", "task", "score"])
    for mdir in sorted(os.listdir(base)):
        model = mdir.replace("__", "/", 1)
        for root, _, files in os.walk(os.path.join(base, mdir)):
            for f in files:
                if not f.endswith(".json") or f == "model_meta.json":
                    continue
                task = f[:-5]
                try:
                    with open(os.path.join(root, f)) as fh:
                        d = json.load(fh)
                except Exception:
                    continue
                score = None
                for split in ("test", "validation", "train"):
                    if split in d.get("scores", {}):
                        entries = d["scores"][split]
                        vals = [e.get("main_score") for e in entries
                                if isinstance(e, dict) and e.get("main_score") is not None]
                        if vals:
                            score = float(np.mean(vals))
                            break
                if score is not None:
                    rows.append(dict(model=model, task=task, score=100 * score))
    return pd.DataFrame(rows)


def boot_corr(x, y, n=10000):
    """Bootstrap over models. Wide intervals at n=12 are the point."""
    x, y = np.asarray(x, float), np.asarray(y, float)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--maeb-cache", default=os.path.join(ROOT, "simteb_audio/maeb_cache"))
    ap.add_argument("--sinhala", default=os.path.join(
        ROOT, "eval/simteb_audio_results_per_task.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/analysis"))
    ap.add_argument("--min-tasks", type=int, default=8,
                    help="a model needs this many MAEB cells to be comparable")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    maeb = load_maeb(a.maeb_cache)
    if maeb.empty:
        print("no MAEB results found"); return
    maeb["domain"] = maeb.task.map(lambda t: DOMAIN.get(t, "?"))

    # Sinhala side: mean over the benchmark, baseline excluded (it is a control)
    sin = pd.read_csv(a.sinhala, sep="\t").set_index("model")
    sin = sin[~sin.index.str.startswith("baseline/")]
    sin_mean = sin.mean(axis=1)

    cov = maeb.pivot_table(index="model", columns="task", values="score")
    counts = cov.notna().sum(axis=1)
    print(f"MAEB coverage: {len(cov)} models x {cov.shape[1]} tasks")
    print(counts.sort_values(ascending=False).to_string())

    # Compare only models with enough MAEB coverage, and use the SAME task set
    # for every model so coverage differences cannot drive the correlation.
    ok = counts[counts >= a.min_tasks].index
    common = [m for m in ok if m in sin_mean.index]
    if len(common) < 4:
        print(f"\nonly {len(common)} comparable models - too few, skipping")
        return
    sub = cov.loc[common]
    shared_tasks = [t for t in sub.columns if sub[t].notna().all()]
    print(f"\ncomparable models: {len(common)}")
    print(f"tasks present for all of them: {len(shared_tasks)}")

    rows = []
    for label, tasks in [
        ("MAEB overall", shared_tasks),
        ("MAEB speech only", [t for t in shared_tasks
                              if DOMAIN.get(t) == "speech"]),
        ("MAEB non-speech", [t for t in shared_tasks
                             if DOMAIN.get(t) in ("music", "environmental")]),
    ]:
        if len(tasks) < 3:
            rows.append(dict(comparison=label, n_tasks=len(tasks),
                             note="too few shared tasks"))
            continue
        m = sub[tasks].mean(axis=1)
        s = sin_mean.loc[common]
        rho = stats.spearmanr(m, s).statistic
        tau = stats.kendalltau(m, s).statistic
        lo, hi = boot_corr(m.values, s.values)
        rows.append(dict(comparison=label, n_models=len(common),
                         n_tasks=len(tasks), spearman=rho, kendall_tau=tau,
                         ci_lo=lo, ci_hi=hi,
                         maeb_best=m.idxmax(), sinhala_best=s.idxmax()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "N_maeb_comparison.tsv"), sep="\t", index=False)

    print("\n## N. MAEB vs Sinhala ranking\n")
    print(df.to_string(index=False))
    print("\n  MAEB mixes music, environmental sound and speech, so only the\n"
          "  speech row speaks to Sinhala transfer. A wide interval at this\n"
          "  number of models means the comparison cannot settle the question,\n"
          "  which is itself the finding.")

    # per-model side by side, useful as a paper table
    side = pd.DataFrame({
        "maeb_overall": sub[shared_tasks].mean(axis=1),
        "maeb_speech": sub[[t for t in shared_tasks
                            if DOMAIN.get(t) == "speech"]].mean(axis=1),
        "sinhala_mean": sin_mean.loc[common],
    })
    for c in side.columns:
        side[c.replace("_", "_rank_", 1) if False else f"rank_{c}"] = \
            side[c].rank(ascending=False, method="min").astype(int)
    side = side.sort_values("rank_sinhala_mean")
    side.to_csv(os.path.join(a.out, "N_maeb_per_model.tsv"), sep="\t")
    print("\nPer model:\n")
    print(side.round(2).to_string())
    print(f"\nwrote {a.out}/N_maeb_comparison.tsv and N_maeb_per_model.tsv")


if __name__ == "__main__":
    main()
