#!/usr/bin/env python3
"""Collect MTEB result files into the two tables the MTEB leaderboard shows.

The leaderboard is not one table but two, both built from the JSON files MTEB
writes to its cache:

  Summary   Rank (Borda) | Model | model metadata | Mean (Task) |
            Mean (TaskType) | one column per TASK TYPE | Release Date
  Per-task  Model | one column per individual TASK

Conventions copied from mteb/leaderboard/table.py and
mteb/benchmarks/_create_table.py:

  * scores are shown x100 with 2 decimals  (_format_scores = round(x*100, 2))
  * models are ranked by Borda count, not by mean: for each task a model earns
    (n_models - rank_in_task) points, summed over tasks. This rewards being
    consistently good rather than winning one task.
  * Mean (Task) averages over tasks; Mean (TaskType) averages the per-type
    means, so a type with many tasks does not dominate.

Writes <out>.tsv (long form), <out>_summary.tsv and <out>_per_task.tsv.
"""

import argparse
import json
import os
import sys

import pandas as pd

# Licences that forbid commercial use. Flagged in the table because a
# benchmark result is only reusable under the model's own terms.
NON_COMMERCIAL = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0",
                  "non-commercial", "llama3", "llama2"}


def model_meta(name):
    """Pull parameters, dim, licence and date from MTEB's own registry."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "simteb_audio", "src"))
        import simteb_audio.models  # noqa: F401  registers our extra models
    except Exception:
        pass
    try:
        import mteb

        m = mteb.get_model_meta(name)
        return dict(
            params=(m.n_parameters / 1e9) if m.n_parameters else None,
            dim=m.embed_dim,
            license=m.license or "",
            date=str(m.release_date) if m.release_date else "",
        )
    except Exception:
        return dict(params=None, dim=None, license="", date="")

# Task type per task, matching the `type` field in each task's metadata.
# The summary table aggregates by type, as the MTEB leaderboard does.
TASK_TYPES = {
    "OpenSLR52SpeakerClustering": "AudioClustering",
    "OpenSLR30SpeakerClustering": "AudioClustering",
    "WorldSpeechSinhalaSessionClustering": "AudioClustering",
    "OpenSLR52SpeakerClassification": "AudioClassification",
    "OpenSLR30SpeakerClassification": "AudioClassification",
    "WorldSpeechSinhalaQualityClassification": "AudioClassification",
    "OpenSLR52SpeakerPairClassification": "AudioPairClassification",
    "OpenSLR30SpeakerPairClassification": "AudioPairClassification",
    "OpenSLR52AudioRetrieval": "AudioRetrieval",
    "OmnilingualSinhalaContentRetrieval": "AudioRetrieval",
    "OmnilingualSinhalaSpeakerClustering": "AudioClustering",
    "VoxLingua107SinhalaVideoClustering": "AudioClustering",
    "OmnilingualSinhalaSpeakerClassification": "AudioClassification",
    "OpenSLR52AudioReranking": "AudioReranking",
    "OpenSLR52A2TRetrieval": "Any2AnyRetrieval",
    "OpenSLR52T2ARetrieval": "Any2AnyRetrieval",
}
TASK_NAMES = {k: k for k in TASK_TYPES}


def fmt(score):
    """MTEB's _format_scores: percentage points, two decimals."""
    return None if pd.isna(score) else round(score * 100, 2)


def walk_results(cache):
    for dirpath, _, filenames in os.walk(cache):
        for fn in filenames:
            if not fn.endswith(".json") or fn.startswith("model_meta"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path) as fh:
                    blob = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if "task_name" not in blob or "scores" not in blob:
                continue

            rel = os.path.relpath(path, cache).split(os.sep)
            if "results" in rel:
                i = rel.index("results")
                model = rel[i + 1] if len(rel) > i + 1 else "unknown"
                revision = rel[i + 2] if len(rel) > i + 2 else ""
            else:
                model = rel[-3] if len(rel) >= 3 else "unknown"
                revision = rel[-2] if len(rel) >= 2 else ""

            task = blob["task_name"]
            for split, entries in blob["scores"].items():
                for entry in entries:
                    yield {
                        "model": model.replace("__", "/"),
                        "revision": revision[:12],
                        "task": TASK_NAMES.get(task, task),
                        "task_type": TASK_TYPES.get(task, "Unknown"),
                        "split": split,
                        "subset": entry.get("hf_subset", "default"),
                        "main_score": entry.get("main_score"),
                        "v_measure": entry.get("v_measure"),
                        "v_measure_std": entry.get("v_measure_std"),
                        "ami": entry.get("ami"),
                        "mteb_version": blob.get("mteb_version", ""),
                        "eval_time_s": blob.get("evaluation_time"),
                    }


def borda_rank(df):
    """MTEB's Borda count: per task a model scores (n_models - rank) points."""
    n_models = df.model.nunique()
    pts = {m: 0.0 for m in df.model.unique()}
    for _, g in df.groupby("task"):
        # average ranks on ties, descending so the best score ranks 1
        ranks = g.set_index("model").main_score.rank(
            method="average", ascending=False)
        for m, r in ranks.items():
            pts[m] += n_models - r
    s = pd.Series(pts)
    return s.rank(method="min", ascending=False).astype(int), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="output prefix, no extension")
    a = ap.parse_args()

    rows = list(walk_results(a.cache))
    if not rows:
        print(f"No MTEB result files found under {a.cache}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    # one score per (model, task): keep the best if a task was rerun
    df = (df.sort_values("main_score")
            .groupby(["model", "task", "task_type"], as_index=False).last())
    df.to_csv(f"{a.out}.tsv", sep="\t", index=False, float_format="%.6f")

    # ---------------- per-task table --------------------------------------
    per_task = df.pivot_table(index="model", columns="task",
                              values="main_score", aggfunc="max")
    per_task = per_task.map(fmt)

    # ---------------- summary table ---------------------------------------
    rank, points = borda_rank(df)
    by_type = (df.groupby(["model", "task_type"]).main_score.mean()
                 .unstack("task_type"))
    summary = pd.DataFrame(index=by_type.index)
    summary.insert(0, "Rank (Borda)", rank)
    meta = {m: model_meta(m) for m in summary.index}
    summary["Total Parameters (B)"] = [meta[m]["params"] for m in summary.index]
    summary["Embedding Dimensions"] = [meta[m]["dim"] for m in summary.index]
    summary["License"] = [meta[m]["license"] for m in summary.index]
    summary["Commercial Use"] = [
        "no" if (meta[m]["license"] or "").lower() in NON_COMMERCIAL else "yes"
        for m in summary.index]
    summary["Mean (Task)"] = df.groupby("model").main_score.mean().map(fmt)
    summary["Mean (TaskType)"] = by_type.mean(axis=1).map(fmt)
    for col in by_type.columns:
        summary[col] = by_type[col].map(fmt)
    summary["Release Date"] = [meta[m]["date"] for m in summary.index]

    summary = summary.sort_values("Rank (Borda)")
    per_task = per_task.loc[summary.index]

    summary.to_csv(f"{a.out}_summary.tsv", sep="\t", float_format="%.2f")
    per_task.to_csv(f"{a.out}_per_task.tsv", sep="\t", float_format="%.2f")

    pd.set_option("display.width", 200)
    print("\nSUMMARY  (scores x100, as the leaderboard shows them)")
    print("=" * 110)
    print(summary.to_string())
    print("\nPER-TASK")
    print("=" * 110)
    print(per_task.to_string())
    print("=" * 110)
    print(f"\n{len(df)} results | {df.model.nunique()} models "
          f"| {df.task.nunique()} tasks | mteb {df.mteb_version.iloc[0]}")
    for p in (f"{a.out}.tsv", f"{a.out}_summary.tsv", f"{a.out}_per_task.tsv"):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
