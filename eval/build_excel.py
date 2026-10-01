#!/usr/bin/env python3
"""Build the results workbook from eval/cluster_scores.tsv.

Two plain sheets: the headline result per model, and the full per-layer table.
Abbreviations are spelled out in a legend beside the results.
"""
import argparse, os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
FONT = "Arial"
BODY = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
UNDER = Border(bottom=Side(style="thin", color="000000"))

MODEL_NAMES = {
    "xlsr300m": "wav2vec2-XLS-R 300M",
    "mms300m": "MMS 300M",
    "w2vbert2": "w2v-BERT 2.0",
    "whisper": "Whisper large-v3",
}

LEGEND = [
    ("ARI", "Adjusted Rand Index. Agreement between the clusters found and the true "
            "speakers, corrected for chance. 0 = random, 1 = perfect."),
    ("NMI", "Normalised Mutual Information. Information shared between the clusters "
            "found and the true speakers. 0 to 1."),
    ("Purity", "Fraction of utterances whose cluster is dominated by their own speaker."),
    ("Silhouette", "How well separated the speaker clusters are, using cosine distance. "
                   "-1 to 1; above 0 means correctly placed."),
    ("Mean cosine", "Average similarity between all pairs of utterances. Near 1 means "
                    "the vectors all point the same way and separate poorly."),
    ("Effective rank", "How many of the dimensions actually carry variation, out of the "
                       "total shown in brackets."),
    ("Content silhouette", "Same as Silhouette, but for groups of utterances that share a "
                           "transcript instead of a speaker."),
    ("DBI", "Davies-Bouldin Index. Cluster spread divided by separation. Lower is better."),
    ("CHI", "Calinski-Harabasz Index. Between-cluster over within-cluster variance. "
            "Higher is better."),
    ("Best layer", "The hidden layer with the highest ARI. Layers are numbered from 0."),
    ("Final layer ARI", "ARI at the model's last layer, for comparison with the best layer."),
]

NOTES = [
    "Each hidden layer is averaged over time into one vector per utterance.",
    "k-means is run with k set to the true number of speakers, without seeing the labels.",
    "Models are frozen: no fine-tuning, no training of any kind.",
    "OpenSLR-52: 11,155 utterances, 476 speakers, 16 kHz.",
    "OpenSLR-30: 2,064 utterances, 12 speakers, 48 kHz resampled to 16 kHz.",
]


def write_row(ws, r, values, font=BODY, formats=None, border=None):
    for i, v in enumerate(values, start=1):
        c = ws.cell(row=r, column=i, value=v)
        c.font = font
        if border:
            c.border = border
        if formats and formats[i - 1]:
            c.number_format = formats[i - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(ROOT, "eval/cluster_scores.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/embedding_results.xlsx"))
    a = ap.parse_args()

    df = pd.read_csv(a.scores, sep="\t")
    wb = Workbook()

    # ------------------------------------------------ Results ----------------
    ws = wb.active
    ws.title = "Results"

    hdr = ["Model", "Dataset", "Speakers", "Best layer", "ARI", "NMI", "Purity",
           "Silhouette", "Mean cosine", "Effective rank", "Final layer ARI"]
    fmts = [None, None, "#,##0", "#,##0", "0.000", "0.000", "0.000",
            "0.000", "0.000", None, "0.000"]
    widths = [22, 13, 10, 11, 8, 8, 8, 10, 12, 14, 14]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.cell(row=1, column=1, value="Speaker clustering of frozen speech embeddings").font = \
        Font(name=FONT, size=11, bold=True)

    r = 3
    write_row(ws, r, hdr, font=BOLD, border=UNDER)
    r += 1
    for ds in ["openslr52", "openslr30"]:
        d = df[df.dataset == ds]
        for mdl, g in sorted(d.groupby("model"),
                             key=lambda kv: -kv[1].ari.max()):
            b = g.loc[g.ari.idxmax()]
            write_row(ws, r, [
                MODEL_NAMES.get(mdl, mdl),
                "OpenSLR-52" if ds == "openslr52" else "OpenSLR-30",
                int(b.k_clusters), int(b.layer),
                round(b.ari, 3), round(b.nmi, 3), round(b.purity, 3),
                round(b.sil_spk, 3), round(b.mean_cos, 3),
                f"{b.eff_rank:.0f} of {int(b.dim)}",
                round(g.sort_values('layer').iloc[-1].ari, 3),
            ], formats=fmts)
            r += 1
        r += 1

    # legend, to the right of the table
    lc = len(hdr) + 2
    ws.column_dimensions[get_column_letter(lc)].width = 20
    ws.column_dimensions[get_column_letter(lc + 1)].width = 72
    lr = 3
    ws.cell(row=lr, column=lc, value="Term").font = BOLD
    ws.cell(row=lr, column=lc, value="Term").border = UNDER
    ws.cell(row=lr, column=lc + 1, value="Meaning").font = BOLD
    ws.cell(row=lr, column=lc + 1, value="Meaning").border = UNDER
    lr += 1
    for term, meaning in LEGEND:
        ws.cell(row=lr, column=lc, value=term).font = BODY
        c = ws.cell(row=lr, column=lc + 1, value=meaning)
        c.font = BODY
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[lr].height = 26
        lr += 1

    lr += 1
    ws.cell(row=lr, column=lc, value="Method").font = BOLD
    lr += 1
    for n in NOTES:
        c = ws.cell(row=lr, column=lc + 1, value=n)
        c.font = BODY
        c.alignment = Alignment(wrap_text=True, vertical="top")
        lr += 1

    ws.freeze_panes = "A4"

    # ------------------------------------------------ All layers -------------
    ws2 = wb.create_sheet("All layers")
    hdr2 = ["Model", "Dataset", "Layer", "ARI", "NMI", "Purity", "Silhouette",
            "Content silhouette", "Mean cosine", "Effective rank", "DBI", "CHI"]
    fmts2 = [None, None, "#,##0", "0.000", "0.000", "0.000", "0.000",
             "0.000", "0.000", "0.0", "0.00", "#,##0"]
    for i, w in enumerate([22, 13, 7, 8, 8, 8, 10, 16, 12, 12, 8, 10], start=1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    write_row(ws2, 1, hdr2, font=BOLD, border=UNDER)
    r = 2
    for ds in ["openslr52", "openslr30"]:
        for mdl in sorted(df.model.unique()):
            g = df[(df.model == mdl) & (df.dataset == ds)].sort_values("layer")
            for _, row in g.iterrows():
                write_row(ws2, r, [
                    MODEL_NAMES.get(mdl, mdl),
                    "OpenSLR-52" if ds == "openslr52" else "OpenSLR-30",
                    int(row.layer), round(row.ari, 3), round(row.nmi, 3),
                    round(row.purity, 3), round(row.sil_spk, 3),
                    round(row.cnt_sil, 3), round(row.mean_cos, 3),
                    round(row.eff_rank, 1), round(row.dbi, 2), int(row.chi),
                ], formats=fmts2)
                r += 1
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = f"A1:{get_column_letter(len(hdr2))}{r-1}"

    wb.save(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
