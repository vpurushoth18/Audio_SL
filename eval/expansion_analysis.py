#!/usr/bin/env python3
"""Run both correlation analyses on whatever checkpoints are currently complete.

Written to be run after every finished checkpoint, so stopping the expansion at
any point leaves a current, self-consistent set of results rather than a
half-written state.

A checkpoint counts only if it has results on BOTH benchmarks. A model present
on one side contributes nothing to a cross-benchmark correlation and is
excluded, with the exclusion recorded rather than silent.

Outputs, all overwritten each run:
  expansion_status.tsv         which checkpoints are in, which are partial
  expansion_scores.tsv         the model x task matrix for complete checkpoints
  expansion_transfer.tsv       MAEB-to-SinAEB, overall and capability-matched
  expansion_task_corr.tsv      task-by-task Spearman within SinAEB
  expansion_group_summary.tsv  same-corpus vs same-capability comparison
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
MIN_SIN, MIN_MAEB = 12, 16          # a checkpoint is usable at this coverage

_sup = open(os.path.join(ROOT, "eval/supplementary.py")).read()
CAPABILITY = dict(re.findall(r'"(\w+)":\s*"(speaker|content|channel)"',
                             re.search(r'CAPABILITY = \{(.*?)\n\}', _sup, re.S).group(1)))
MAEB_CAP = dict(re.findall(r'"(\w+)":\s*"(speaker|content|non-speech)"',
                           re.search(r'MAEB_CAP = \{(.*?)\n\}', _sup, re.S).group(1)))
CORPUS = {t: ("OpenSLR-52" if t.startswith("OpenSLR52") else
              "OpenSLR-30" if t.startswith("OpenSLR30") else
              "Omnilingual" if t.startswith("Omnilingual") else
              "WorldSpeech" if t.startswith("WorldSpeech") else "VoxLingua107")
          for t in CAPABILITY}


def scrape(cache):
    rows = []
    base = os.path.join(cache, "results")
    if not os.path.isdir(base):
        return pd.DataFrame(columns=["model", "task", "score"])
    for mdir in sorted(os.listdir(base)):
        model = mdir.replace("__", "/", 1)
        for f in glob.glob(os.path.join(base, mdir, "**", "*.json"), recursive=True):
            task = os.path.basename(f)[:-5]
            if task == "model_meta":
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for _, entries in d.get("scores", {}).items():
                for e in entries:
                    if isinstance(e, dict) and e.get("main_score") is not None:
                        rows.append(dict(model=model, task=task,
                                         score=float(e["main_score"])))
                        break
                break
    return pd.DataFrame(rows)


def ci(x, y, n=4000):
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        out.append(stats.spearmanr(x[i], y[i]).statistic)
    return (float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def main():
    sin = scrape(os.path.join(ROOT, "simteb_audio/mteb_cache"))
    maeb = scrape(os.path.join(ROOT, "simteb_audio/maeb_cache"))
    sin = sin[sin.task.isin(CAPABILITY)]

    ns = sin.groupby("model").task.nunique()
    nm = maeb.groupby("model").task.nunique()
    allm = sorted(set(ns.index) | set(nm.index))
    status = pd.DataFrame({
        "model": allm,
        "sinaeb_tasks": [int(ns.get(m, 0)) for m in allm],
        "maeb_tasks": [int(nm.get(m, 0)) for m in allm]})
    status["complete"] = ((status.sinaeb_tasks >= MIN_SIN) &
                          (status.maeb_tasks >= MIN_MAEB))
    status["is_baseline"] = status.model.str.startswith("baseline/")
    status = status.sort_values(["complete", "model"], ascending=[False, True])
    status.to_csv(os.path.join(A, "expansion_status.tsv"), sep="\t", index=False)

    usable = status[(status.complete) & (~status.is_baseline)].model.tolist()
    M = len(usable)
    print(f"  usable checkpoints (both benchmarks): {M}")
    partial = status[(~status.complete) & (~status.is_baseline)]
    if len(partial):
        print(f"  partial, excluded: {len(partial)} -> "
              f"{', '.join(partial.model.head(4))}" + (" ..." if len(partial) > 4 else ""))
    if M < 4:
        print("  fewer than 4 usable checkpoints; correlations need more. stopping here.")
        return

    W = sin[sin.model.isin(usable)].pivot_table(index="model", columns="task",
                                                values="score")
    C = maeb[maeb.model.isin(usable)].pivot_table(index="model", columns="task",
                                                  values="score") * 100
    W.to_csv(os.path.join(A, "expansion_scores.tsv"), sep="\t", float_format="%.17g")

    # ---------- cross-benchmark transfer
    shared = [t for t in C.columns if C[t].notna().all()]
    cap_mean = {c: W[[t for t in W.columns if CAPABILITY.get(t) == c]].mean(axis=1)
                for c in ("speaker", "content")}
    rows = []
    for mcap, tasks in [("all", shared)] + [
            (c, [t for t in shared if MAEB_CAP.get(t) == c])
            for c in ("speaker", "content", "non-speech")]:
        if len(tasks) < 3:
            continue
        x = C[tasks].mean(axis=1)
        for scap in ("speaker", "content"):
            y = cap_mean[scap]
            lo, hi = ci(x.values, y.values)
            rows.append(dict(maeb_subset=mcap, n_maeb_tasks=len(tasks),
                             sinaeb_capability=scap, n_checkpoints=M,
                             spearman=stats.spearmanr(x, y).statistic,
                             ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0)))
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join(A, "expansion_transfer.tsv"), sep="\t", index=False,
              float_format="%.17g")
    print(f"\n  transfer ({M} checkpoints):")
    print(tr[["maeb_subset", "sinaeb_capability", "spearman", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------- task-by-task within SinAEB
    tasks = list(W.columns)
    rows = []
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            x, y = W[a].values, W[b].values
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            lo, hi = ci(x, y)
            sc, scap = CORPUS[a] == CORPUS[b], CAPABILITY[a] == CAPABILITY[b]
            rows.append(dict(task_a=a, task_b=b,
                             corpus_a=CORPUS[a], corpus_b=CORPUS[b],
                             capability_a=CAPABILITY[a], capability_b=CAPABILITY[b],
                             same_corpus=sc, same_capability=scap,
                             group=("same corpus, same capability" if sc and scap
                                    else "same corpus, different capability" if sc
                                    else "different corpus, same capability" if scap
                                    else "different corpus, different capability"),
                             spearman=stats.spearmanr(x, y).statistic,
                             ci_lo=lo, ci_hi=hi, n_checkpoints=M))
    tc = pd.DataFrame(rows)
    tc.to_csv(os.path.join(A, "expansion_task_corr.tsv"), sep="\t", index=False,
              float_format="%.17g")
    g = tc.groupby("group").spearman.agg(["count", "median", "mean"])
    g.to_csv(os.path.join(A, "expansion_group_summary.tsv"), sep="\t",
             float_format="%.17g")
    print(f"\n  task pairs ({len(tc)} pairs, {M} checkpoints):")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))

    a = tc[tc.group == "same corpus, different capability"].spearman
    b = tc[tc.group == "different corpus, same capability"].spearman
    if len(a) and len(b):
        print(f"\n  same corpus/diff capability  : {a.median():+.3f} (n={len(a)})")
        print(f"  diff corpus/same capability  : {b.median():+.3f} (n={len(b)})")
        print(f"  difference                   : {a.median() - b.median():+.3f}")

    with open(os.path.join(A, "expansion_state.json"), "w") as f:
        json.dump(dict(usable_checkpoints=M, usable=usable,
                       partial=partial.model.tolist(),
                       updated=pd.Timestamp.now().isoformat()), f, indent=2)
    print(f"\n  wrote expansion_{{status,scores,transfer,task_corr,group_summary}}.tsv")


if __name__ == "__main__":
    main()
