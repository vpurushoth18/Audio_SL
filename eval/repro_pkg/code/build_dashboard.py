#!/usr/bin/env python3
# Builds eval/dashboard.html from the tables written by collect_mteb_results.py.
#
#   python eval/collect_mteb_results.py --cache simteb_audio/mteb_cache \
#       --out eval/simteb_audio_results
#   python eval/build_dashboard.py

import argparse
import html
import os

import pandas as pd

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"

# column headers for the per-task table
SHORT = {
    "OpenSLR52SpeakerClustering": "SLR52 spk clust",
    "OpenSLR30SpeakerClustering": "SLR30 spk clust",
    "WorldSpeechSinhalaSessionClustering": "WS session clust",
    "OpenSLR52SpeakerClassification": "SLR52 spk clf",
    "OpenSLR30SpeakerClassification": "SLR30 spk clf",
    "WorldSpeechSinhalaQualityClassification": "WS quality clf",
    "OpenSLR52SpeakerPairClassification": "SLR52 spk pair",
    "OpenSLR30SpeakerPairClassification": "SLR30 spk pair",
    "OpenSLR52AudioRetrieval": "SLR52 retrieval",
    "OpenSLR52AudioReranking": "SLR52 rerank",
}
METRIC = {
    "AudioClustering": "V-measure",
    "AudioClassification": "accuracy",
    "AudioPairClassification": "max AP",
    "AudioRetrieval": "hit rate@5",
    "AudioReranking": "MAP@5",
}

CSS = """
body { font-family: Georgia, serif; max-width: 1100px; margin: 30px auto;
       padding: 0 15px; color: #222; background: #fff; }
h1 { font-size: 24px; }
h2 { font-size: 18px; margin-top: 30px; border-bottom: 1px solid #ccc; }
table { border-collapse: collapse; font-family: Arial, sans-serif; font-size: 13px; }
th, td { border: 1px solid #bbb; padding: 4px 8px; }
th { background: #eee; }
td.num { text-align: right; }
p.small { font-size: 13px; color: #555; }
"""


def shade(v, lo, hi):
    # light-to-mid blue, relative to the column's range
    if hi - lo < 1e-9:
        t = 0.5
    else:
        t = (v - lo) / (hi - lo)
    g = int(240 - 110 * t)
    return f"background: rgb({g}, {g + 10}, 255)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "eval/simteb_audio_results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/dashboard.html"))
    a = ap.parse_args()

    summary = pd.read_csv(f"{a.results}_summary.tsv", sep="\t")
    per_task = pd.read_csv(f"{a.results}_per_task.tsv", sep="\t")
    long = pd.read_csv(f"{a.results}.tsv", sep="\t")

    tasks = [c for c in per_task.columns if c != "model"]
    task_type = long.drop_duplicates("task").set_index("task").task_type.to_dict()
    summary = summary.sort_values("Rank (Borda)")
    per_task = per_task.set_index("model").loc[summary.model]

    out = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
           "<title>SiMTEB-Audio results</title>", f"<style>{CSS}</style></head><body>",
           "<h1>SiMTEB-Audio results</h1>",
           f"<p>{len(summary)} models, {len(tasks)} tasks. Scores are x100. "
           f"Last updated {pd.Timestamp.now():%Y-%m-%d}.</p>"]

    # overall table
    out.append("<h2>Overall</h2>")
    out.append("<p class='small'>Sorted by Borda rank. Borda rank and mean score "
               "don't always agree (e.g. xls-r-2b has the highest mean but does "
               "badly on retrieval).</p>")
    cols = ["Rank (Borda)", "model", "Total Parameters (B)", "Embedding Dimensions",
            "License", "Mean (Task)", "Mean (TaskType)"]
    heads = ["Rank", "Model", "Params (B)", "Dim", "License", "Mean (task)", "Mean (type)"]
    out.append("<table><tr>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr>")
    for _, r in summary.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "License" and r["Commercial Use"] == "no":
                v = f"{v} (non-commercial)"
            if isinstance(v, float):
                cells.append(f"<td class='num'>{v:.2f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")

    # per-task table
    out.append("<h2>Per task</h2>")
    out.append("<p class='small'>Shading is per column (darker = better), since the "
               "metric changes with the task type. Best score in each column is in bold.</p>")
    out.append("<table><tr><th>Model</th>" + "".join(
        f"<th title='{t}'>{SHORT.get(t, t)}<br><small>{METRIC.get(task_type.get(t), '')}</small></th>"
        for t in tasks) + "</tr>")
    for m, row in per_task.iterrows():
        cells = [f"<td>{html.escape(m)}</td>"]
        for t in tasks:
            v = row[t]
            if pd.isna(v):
                cells.append("<td class='num'>-</td>")
                continue
            col = per_task[t].dropna()
            txt = f"{v:.1f}"
            if v == col.max():
                txt = f"<b>{txt}</b>"
            cells.append(f"<td class='num' style='{shade(v, col.min(), col.max())}'>{txt}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")

    out.append("<p class='small'>Note: WS session clustering gives almost the same score "
               "(~91-93) for every model, so I don't think it tells us much. V-measure "
               "isn't chance-corrected and there are 84 clusters.</p>")
    out.append("</body></html>")

    with open(a.out, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
