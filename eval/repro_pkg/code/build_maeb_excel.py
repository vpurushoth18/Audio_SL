#!/usr/bin/env python3
"""Turn eval/maeb_results.tsv into the MAEB deliverable workbook.

Six sheets:
  Read Me       what the whole evaluation is, in plain language
  Ranking       Borda count and per-category means, the MAEB leaderboard view
  Per Task      dataset x task matrix of main metrics, acoustic vs linguistic
  Acoustic vs Linguistic   the trade-off the experiment was built to test
  Clustering    every clustering score next to its random-partition baseline
  Methodology   protocol, deviations, and what not to trust

Every sheet ends with a LEGEND defining each short name used on it. Derived
cells are Excel formulas, not baked values, so the sheet recalculates if a
score is corrected.
"""
import argparse, os
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1F3864")
SUB_FILL = PatternFill("solid", fgColor="D9E2F3")
LEG_FILL = PatternFill("solid", fgColor="F2F2F2")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BODY = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=11, bold=True, color="1F3864")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
BEST_FILL = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
DEC4, DEC3, INT = "0.0000", "0.000", "#,##0"

ACOUSTIC = ("Speaker", "Corpus")

DATASETS = {
    "openslr30":    "OpenSLR30 (SLR30, 48 kHz, 12 spk)",
    "openslr52":    "OpenSLR52 (SLR52, 16 kHz, 476 spk)",
    "openslr30+52": "both corpora (label = corpus)",
}

# ---------------------------------------------------------------- legends
MODELS_LEG = [
    ("mms300m", "facebook/mms-300m",
     "Massively Multilingual Speech, 300M parameters. Self-supervised on raw "
     "audio in 1,400+ languages; never shown a transcript during pretraining."),
    ("w2vbert2", "facebook/w2v-bert-2.0",
     "Wav2Vec-BERT 2.0, 580M parameters. Self-supervised on 4.5M hours of "
     "unlabelled audio; the speech encoder used inside SeamlessM4T."),
    ("whisper", "openai/whisper-large-v3 (encoder only)",
     "Trained to TRANSCRIBE speech from 680k hours of labelled audio. The only "
     "model here that saw text during training, which is why it behaves "
     "differently on the content tasks."),
    ("xlsr300m", "facebook/wav2vec2-xls-r-300m",
     "Cross-Lingual Speech Representations, 300M parameters. Self-supervised on "
     "raw audio in 128 languages."),
]

DATA_LEG = [
    ("SLR30 / OpenSLR30", "OpenSLR resource 30, Sinhala TTS corpus",
     "2,064 clips, 12 speakers, 48 kHz studio recordings. Only 26 clips share a "
     "transcript with another speaker, so this corpus supports speaker tasks only."),
    ("SLR52 / OpenSLR52", "OpenSLR resource 52, Large Sinhala ASR corpus",
     "11,155 clips, 476 speakers, 16 kHz. Has 4,000 groups of clips where "
     "different speakers read the SAME sentence, which is what makes the "
     "content tasks possible."),
    ("spk", "speakers", "Number of distinct people recorded in the corpus."),
    ("kHz", "kilohertz, the sampling rate",
     "How finely the audio was digitised. SLR30 is 48 kHz and SLR52 is 16 kHz, "
     "so the two corpora sound measurably different and must never be pooled."),
]

TASK_LEG = [
    ("Classification", "Classification",
     "Train a simple classifier (logistic regression) on just 8 example clips "
     "per label, then ask it to label unseen clips. Score = fraction correct."),
    ("Clustering", "Clustering",
     "Hide the labels. Let k-means group the vectors into k blobs on its own. "
     "Then check whether those blobs happen to match the real labels."),
    ("PairClassification", "Pair Classification",
     "Take 20,000 pairs of clips. Measure the distance within each pair. Do the "
     "pairs that SHOULD match sit closer together than the pairs that shouldn't?"),
    ("A2ARetrieval", "Audio-to-Audio Retrieval",
     "Take one clip. Sort every other clip by distance from it. Is a correct "
     "match inside the top 5? Repeated for 2,000 query clips."),
    ("Reranking", "Reranking",
     "Same as retrieval, but ranking only a short pre-picked candidate list that "
     "has been deliberately stuffed with hard distractors."),
    ("Speaker*", "Speaker tasks (the ACOUSTIC family)",
     "The label is WHO is speaking. Two clips match if the same person said them."),
    ("Content*", "Content tasks (the LINGUISTIC family)",
     "The label is WHAT was said. Two clips match if they are the same sentence "
     "read by DIFFERENT people. Clips from the query's own speaker are removed "
     "from the ranking first, so a model cannot score by recognising the voice."),
    ("Corpus_Classification", "Corpus Classification",
     "The only task using both corpora: predict which corpus a clip came from. "
     "Largely a recording-quality test, included for completeness."),
]

METRIC_LEG = [
    ("accuracy", "Accuracy", "Fraction of clips labelled correctly. 0 to 1, higher is better."),
    ("v_measure", "V-measure",
     "How well the k-means blobs line up with the true labels. 0 to 1, higher is "
     "better. WARNING: not corrected for chance, so it must always be read "
     "against the random baseline column on the Clustering sheet."),
    ("ARI", "Adjusted Rand Index",
     "The same idea as V-measure but corrected for chance: random guessing "
     "scores 0. This is the trustworthy clustering number."),
    ("max_ap", "Maximum Average Precision",
     "Rank all pairs by similarity; measures whether true matching pairs sit "
     "above non-matching ones. 0 to 1, 0.5 is roughly chance on balanced pairs."),
    ("cv_recall_at_5", "Cross-Validated Recall at 5",
     "Fraction of query clips whose top-5 nearest neighbours contain at least "
     "one correct match. 0 to 1, higher is better."),
    ("map_at_1000", "Mean Average Precision at 1000",
     "Average quality of the ranking over the top 1,000 candidates, rewarding "
     "correct matches placed near the top. 0 to 1, higher is better."),
    ("random / baseline", "Random-partition baseline",
     "The same metric computed after shuffling the labels. Any real score must "
     "beat this or it is measuring nothing."),
    ("k", "k, the number of clusters", "How many groups k-means was asked to find, set to the true label count."),
    ("n", "n, the number of points", "How many clips went into that task."),
]

GENERAL_LEG = [
    ("MAEB", "Massive Audio Embedding Benchmark",
     "The paper being replicated: arXiv:2602.16008v1 (Feb 2026)."),
    ("embedding / vector", "Audio embedding",
     "Each audio clip is run through a frozen model and comes out as one list of "
     "~1,024 numbers. All scoring compares distances between these."),
    ("frozen", "Frozen model",
     "The models are used exactly as downloaded. Nothing is trained or "
     "fine-tuned on Sinhala at any point."),
    ("Borda", "Borda count",
     "The ranking method MAEB uses. Each task ranks the 4 models; a model earns "
     "(4 - its rank) points per task; points are summed over all 13 tasks."),
    ("acoustic", "Acoustic tasks",
     "Tasks whose label is WHO is speaking (or which corpus it came from)."),
    ("linguistic", "Linguistic tasks",
     "Tasks whose label is WHAT was said."),
    ("A2A", "Audio-to-Audio", "Both the query and the search corpus are audio."),
    ("Clf / PC / Rrnk / Rtrvl", "Classification / Pair Classification / Reranking / Retrieval",
     "Short column names for the five task categories."),
]


def header(ws, cols, freeze="B2"):
    for i, (name, width) in enumerate(cols, start=1):
        c = ws.cell(row=1, column=i, value=name)
        c.fill, c.font = HDR_FILL, HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = freeze
    ws.row_dimensions[1].height = 30


def put(ws, r, c, v, fmt=None, font=BODY, fill=None):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font, cell.border = font, BORDER
    if fmt:
        cell.number_format = fmt
    if fill:
        cell.fill = fill
    return cell


def legend(ws, row, blocks, note=None):
    """Write 'what the short names mean' blocks at the bottom of a sheet."""
    r = row + 2
    ws.cell(row=r, column=1, value="LEGEND - full form and meaning of every "
            "short name used on this sheet").font = TITLE
    r += 1
    if note:
        c = ws.cell(row=r, column=1, value=note)
        c.font = BODY
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 28
        r += 2
    for title, items in blocks:
        ws.cell(row=r, column=1, value=title).font = BOLD
        r += 1
        for a, b, c_ in (("short name", "full form", "what it means"),):
            for i, v in enumerate((a, b, c_), start=1):
                cell = ws.cell(row=r, column=i, value=v)
                cell.font, cell.fill, cell.border = BOLD, LEG_FILL, BORDER
        r += 1
        for short, full, desc in items:
            for i, v in enumerate((short, full, desc), start=1):
                cell = ws.cell(row=r, column=i, value=v)
                cell.font, cell.border = BODY, BORDER
                cell.alignment = Alignment(wrap_text=(i == 3), vertical="top")
            ws.row_dimensions[r].height = max(14, 13 * (1 + len(desc) // 95))
            r += 1
        r += 1
    return r


def widen_for_legend(ws, ncols):
    """Legend uses columns A-C; make sure they can hold it."""
    for col, w in (("A", 30), ("B", 42), ("C", 95)):
        cur = ws.column_dimensions[col].width or 0
        ws.column_dimensions[col].width = max(cur, w)


def probes_of(task):
    return "acoustic" if any(k in task for k in ACOUSTIC) else "linguistic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "eval/maeb_results.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/maeb_eval.xlsx"))
    a = ap.parse_args()

    df = pd.read_csv(a.results, sep="\t")
    models = sorted(df.model.unique())
    piv = df.pivot(index="task", columns="model", values="score")
    info = (df.drop_duplicates("task").set_index("task")[["dataset", "main_metric"]]
              .to_dict("index"))
    tasks = sorted(piv.index, key=lambda t: (info[t]["dataset"], probes_of(t), t))

    wb = Workbook()

    # ================= Read Me ==============================================
    rm = wb.active
    rm.title = "Read Me"
    rm.column_dimensions["A"].width = 30
    rm.column_dimensions["B"].width = 112
    story = [
        ("What this workbook is", ""),
        ("In one sentence",
         "Four speech models were used to turn 13,219 Sinhala audio clips into "
         "numeric vectors, and these sheets measure whether clips that SHOULD be "
         "similar actually ended up close together."),
        ("", ""),
        ("The method, step by step", ""),
        ("1. Audio to vectors",
         "Each of 13,219 Sinhala clips is passed through a frozen model and comes "
         "out as one vector of about 1,024 numbers. Four models = four sets of "
         "vectors over the same clips. No model is trained or fine-tuned."),
        ("2. Two ideas of 'similar'",
         "The same clips carry TWO different labels. speaker_id = who is talking. "
         "content_group = clips of the SAME sentence read by DIFFERENT people. "
         "Take clips A and B by one speaker and clip C of A's sentence read by "
         "someone else: by voice A-B match, by words A-C match. A model cannot "
         "put both pairs closest, so whichever it prefers reveals what it encodes."),
        ("3. Five ways to score",
         "The paper defines five evaluators - Classification, Clustering, Pair "
         "Classification, Retrieval, Reranking. Each is a different formula for "
         "turning those distances into one number. See the legend below."),
        ("4. The anti-cheat rule",
         "In every content task, clips by the query's OWN speaker are deleted from "
         "the candidate list before ranking. Without this a model scores well just "
         "by matching the voice, since a speaker's own clips are already nearby, "
         "and the score would say nothing about understanding the words."),
        ("", ""),
        ("What was found", ""),
        ("The headline",
         "On 'find the same sentence spoken by a different person' (top-5), "
         "w2vbert2 scores 0.0070 and mms300m 0.0060, where random guessing is "
         "about 0.001 - effectively blind to the words. The same two models score "
         "0.73 and 0.66 at finding the same VOICE. whisper is the mirror image: "
         "0.26 at voices, 0.25 at sentences, the only model above noise on words. "
         "Self-supervised audio models encode WHO; whisper, trained to transcribe, "
         "encodes WHAT."),
        ("", ""),
        ("Where to look", ""),
        ("Ranking", "Overall leaderboard across all 13 tasks."),
        ("Per Task", "Every score, one row per dataset and task."),
        ("Acoustic vs Linguistic", "The who-versus-what trade-off, per model."),
        ("Clustering", "Clustering scores beside their random baselines. Read this "
                       "before quoting any clustering number."),
        ("Methodology", "Full protocol, deviations from the paper, and limitations."),
    ]
    r = 1
    for k, v in story:
        ca = rm.cell(row=r, column=1, value=k)
        cb = rm.cell(row=r, column=2, value=v)
        ca.font = TITLE if v == "" else BOLD
        cb.font = BODY
        cb.alignment = Alignment(wrap_text=True, vertical="top")
        ca.alignment = Alignment(vertical="top")
        rm.row_dimensions[r].height = max(14, 13 * (1 + len(v) // 105))
        r += 1
    widen_for_legend(rm, 3)
    rm.column_dimensions["B"].width = 112
    legend(rm, r, [("Models evaluated", MODELS_LEG),
                   ("Datasets", DATA_LEG),
                   ("Task types", TASK_LEG),
                   ("Metrics", METRIC_LEG),
                   ("General terms", GENERAL_LEG)])

    # ================= Per Task =============================================
    ws = wb.create_sheet("Per Task")
    cols = ([("dataset", 34), ("task", 30), ("probes", 11), ("main metric", 15)]
            + [(m, 12) for m in models])
    header(ws, cols, freeze="E2")
    rows_for, r = {}, 2
    for key in tasks:
        ds, probes = info[key]["dataset"], probes_of(key)
        put(ws, r, 1, DATASETS.get(ds, ds))
        put(ws, r, 2, key.split("_", 1)[-1], font=BOLD)
        put(ws, r, 3, probes, fill=SUB_FILL if probes == "linguistic" else None)
        put(ws, r, 4, info[key]["main_metric"])
        vals = [piv.loc[key, m] for m in models]
        best = max(vals)
        for j, v in enumerate(vals, start=5):
            put(ws, r, j, float(v), DEC4, fill=BEST_FILL if v == best else None)
        rows_for[key] = r
        r += 1
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{r-1}"

    first_col, last_col = 5, 4 + len(models)
    for ds in sorted({info[k]["dataset"] for k in tasks}):
        for want in ("acoustic", "linguistic"):
            refs = [rows_for[k] for k in tasks
                    if info[k]["dataset"] == ds and probes_of(k) == want]
            if not refs:
                continue
            put(ws, r, 1, DATASETS.get(ds, ds), font=BOLD)
            put(ws, r, 2, f"mean ({want})", font=BOLD)
            put(ws, r, 3, want, fill=SUB_FILL if want == "linguistic" else None)
            put(ws, r, 4, f"{len(refs)} tasks")
            for j in range(first_col, last_col + 1):
                L = get_column_letter(j)
                put(ws, r, j, f"=AVERAGE({','.join(f'{L}{x}' for x in refs)})",
                    DEC4, font=BOLD)
            r += 1
    widen_for_legend(ws, len(cols))
    legend(ws, r, [("Datasets", DATA_LEG), ("Task types", TASK_LEG),
                   ("Metrics", METRIC_LEG), ("Models", MODELS_LEG)],
           note="Green cell = best model on that task. Blue 'probes' cell = a "
                "linguistic task. Mean rows are computed PER DATASET; the two "
                "corpora are never averaged together because their recording "
                "conditions differ.")

    # ================= Acoustic vs Linguistic ===============================
    av = wb.create_sheet("Acoustic vs Linguistic")
    acols = [("dataset", 34), ("model", 14), ("acoustic mean", 14),
             ("linguistic mean", 15), ("gap (ling - acou)", 17),
             ("leans", 22), ("n acoustic", 11), ("n linguistic", 12)]
    header(av, acols, freeze="C2")
    ar = 2
    for ds in sorted(df.dataset.unique()):
        sub_ds = df[df.dataset == ds]
        na = sub_ds[sub_ds.task.map(probes_of) == "acoustic"].task.nunique()
        nl = sub_ds[sub_ds.task.map(probes_of) == "linguistic"].task.nunique()
        for m in models:
            d_m = sub_ds[sub_ds.model == m]
            acou = d_m[d_m.task.map(probes_of) == "acoustic"].score
            ling = d_m[d_m.task.map(probes_of) == "linguistic"].score
            put(av, ar, 1, DATASETS.get(ds, ds))
            put(av, ar, 2, m, font=BOLD)
            put(av, ar, 3, float(acou.mean()) if len(acou) else None, DEC4)
            put(av, ar, 4, float(ling.mean()) if len(ling) else None, DEC4)
            if len(acou) and len(ling):
                put(av, ar, 5, f"=D{ar}-C{ar}", DEC4)
                put(av, ar, 6, f'=IF(E{ar}>0.05,"leans linguistic",'
                               f'IF(E{ar}<-0.05,"leans acoustic","balanced"))')
            else:
                put(av, ar, 5, None)
                put(av, ar, 6, "n/a - one family only")
            put(av, ar, 7, na, INT)
            put(av, ar, 8, nl, INT)
            ar += 1
    widen_for_legend(av, len(acols))
    legend(av, ar, [("Terms on this sheet", [
        ("acoustic mean", "Mean of the acoustic (speaker) tasks",
         "Average score over tasks whose label is WHO is speaking."),
        ("linguistic mean", "Mean of the linguistic (content) tasks",
         "Average score over tasks whose label is WHAT was said."),
        ("gap (ling - acou)", "Gap = linguistic mean minus acoustic mean",
         "Positive = the model encodes the WORDS better than the VOICE. "
         "Negative = the reverse. This single number is the trade-off the paper "
         "predicts and this experiment was built to test."),
        ("leans", "Which way the model leans",
         "'leans linguistic' if the gap is above +0.05, 'leans acoustic' if below "
         "-0.05, otherwise 'balanced'."),
        ("n acoustic / n linguistic", "Number of tasks in each family",
         "Shows why some rows cannot be compared: OpenSLR30 has 4 acoustic tasks "
         "and 0 linguistic ones, so no trade-off can be computed for it."),
        ("n/a - one family only", "Not applicable",
         "That dataset has tasks of only one family, so no gap exists. Only "
         "OpenSLR52 carries both and can test the trade-off.")]),
        ("Models", MODELS_LEG), ("Datasets", DATA_LEG)])

    # ================= Ranking ==============================================
    s = wb.create_sheet("Ranking")
    dslist = sorted(df.dataset.unique())
    scols = ([("model", 14), ("MAEB rank", 11), ("Borda points", 13), ("mean score", 12),
              ("Classification", 13), ("Clustering", 12), ("Pair Clf", 11),
              ("Reranking", 12), ("Retrieval", 11)]
             + [(f"mean: {DATASETS.get(d, d).split(' (')[0]}", 18) for d in dslist])
    header(s, scols, freeze="B2")
    nm = len(models)
    borda = piv.rank(axis=1, ascending=False, method="average").rsub(nm + 1).sum(axis=0)
    order = borda.sort_values(ascending=False).index.tolist()
    cat = {t: df[df.task_type == t].pivot(index="task", columns="model", values="score")
           for t in df.task_type.unique()}
    sr = 2
    for m in order:
        put(s, sr, 1, m, font=BOLD)
        put(s, sr, 2, sr - 1, INT)
        put(s, sr, 3, float(borda[m]), DEC3)
        put(s, sr, 4, float(piv[m].mean()), DEC4)
        for j, t in enumerate(["classification", "clustering", "pair",
                               "reranking", "retrieval"], start=5):
            put(s, sr, j, float(cat[t][m].mean()) if t in cat else None, DEC4)
        for j, d in enumerate(dslist, start=10):
            sel = df[(df.dataset == d) & (df.model == m)].score
            put(s, sr, j, float(sel.mean()) if len(sel) else None, DEC4)
        sr += 1
    widen_for_legend(s, len(scols))
    legend(s, sr, [("Columns on this sheet", [
        ("MAEB rank", "Rank on this benchmark", "1 = best, by Borda points."),
        ("Borda points", "Borda count score",
         "Each of the 13 tasks ranks the 4 models; a model earns (4 - its rank) "
         "points per task, summed. Max possible = 13 tasks x 3 points = 39."),
        ("mean score", "Arithmetic mean of all 13 task scores",
         "Shown next to Borda because the two can disagree - they rank different "
         "models first when one model wins a few tasks by a wide margin."),
        ("Pair Clf", "Pair Classification", "Mean over the pair-classification tasks."),
        ("mean: <corpus>", "Mean score on that corpus",
         "Average of that model's scores on tasks built from that dataset."),
    ]), ("Task categories", TASK_LEG), ("Metrics", METRIC_LEG),
        ("Models", MODELS_LEG), ("General terms", GENERAL_LEG)])

    # ================= Clustering ===========================================
    cl = df[df.task_type == "clustering"][
        ["model", "dataset", "task", "m_v_measure", "m_rand_v_measure", "m_ari",
         "m_rand_ari", "m_k", "m_n"]].copy()
    c = wb.create_sheet("Clustering")
    ccols = [("dataset", 34), ("task", 26), ("model", 14), ("V-measure", 12),
             ("random V-measure", 17), ("lift over random", 16), ("ARI", 11),
             ("random ARI", 12), ("k (classes)", 11), ("n (points)", 11),
             ("verdict", 22)]
    header(c, ccols, freeze="D2")
    cr = 2
    for _, row in cl.sort_values(["task", "m_v_measure"], ascending=[True, False]).iterrows():
        below = row.m_v_measure <= row.m_rand_v_measure
        fill = WARN_FILL if below else None
        put(c, cr, 1, DATASETS.get(row.dataset, row.dataset), fill=fill)
        put(c, cr, 2, row.task.split("_", 1)[-1], fill=fill)
        put(c, cr, 3, row.model, font=BOLD, fill=fill)
        put(c, cr, 4, float(row.m_v_measure), DEC4, fill=fill)
        put(c, cr, 5, float(row.m_rand_v_measure), DEC4, fill=fill)
        put(c, cr, 6, f"=D{cr}-E{cr}", DEC4, fill=fill)
        put(c, cr, 7, float(row.m_ari), DEC4, fill=fill)
        put(c, cr, 8, float(row.m_rand_ari), DEC4, fill=fill)
        put(c, cr, 9, int(row.m_k), INT, fill=fill)
        put(c, cr, 10, int(row.m_n), INT, fill=fill)
        put(c, cr, 11, f'=IF(D{cr}<=E{cr},"AT OR BELOW CHANCE",'
                       f'IF(G{cr}<0.05,"weak","real signal"))', fill=fill)
        cr += 1
    widen_for_legend(c, len(ccols))
    legend(c, cr, [("Columns on this sheet", [
        ("V-measure", "V-measure",
         "Agreement between the k-means blobs and the true labels, 0 to 1. NOT "
         "chance-corrected, so it is meaningless on its own."),
        ("random V-measure", "V-measure of a randomly shuffled partition",
         "The same metric after assigning every clip to a random cluster. This is "
         "the number a model must beat."),
        ("lift over random", "V-measure minus random V-measure",
         "How much better than shuffling. Zero or negative means no signal."),
        ("ARI", "Adjusted Rand Index",
         "Chance-corrected agreement: random scores 0. The number to trust."),
        ("k (classes)", "k, the number of clusters",
         "Set to the true number of labels, as the paper specifies."),
        ("n (points)", "n, the number of clips", "How many clips entered the task."),
        ("verdict", "Automatic reading of the row",
         "'AT OR BELOW CHANCE' (amber) = the score does not beat shuffling and "
         "must not be quoted. 'weak' = beats chance but ARI under 0.05. "
         "'real signal' = genuinely above chance."),
    ]), ("Datasets", DATA_LEG), ("Models", MODELS_LEG)],
        note="Amber rows failed the random-baseline check. All four "
             "ContentClustering rows are amber or near it: the content groups in "
             "this data are too small (mostly 2-3 clips) for clustering to work, "
             "so content ability is measured by Retrieval and Reranking instead.")

    # ================= Methodology ==========================================
    m = wb.create_sheet("Methodology")
    m.column_dimensions["A"].width = 30
    m.column_dimensions["B"].width = 108
    notes = [
        ("MAEB replication", ""),
        ("Source",
         "MAEB: Massive Audio Embedding Benchmark, arXiv:2602.16008v1 (Feb 2026). "
         "This workbook reproduces the paper's EVALUATION PROTOCOL on Sinhala "
         "speech. It does not reproduce the paper's own tasks, so scores are "
         "comparable across the four models here, NOT against the published "
         "leaderboard."),
        ("Protocol",
         "Every model is a frozen encoder. No fine-tuning. Audio is mean-pooled "
         "over time from the FINAL transformer layer into one vector per clip, "
         "L2-normalised, and all tasks run on those vectors using cosine distance."),
        ("Datasets evaluated",
         "OpenSLR52: 11,155 clips, 476 speakers, 16 kHz, 4,000 cross-speaker "
         "transcript groups. OpenSLR30: 2,064 clips, 12 speakers, 48 kHz, only 26 "
         "clips in 13 transcript groups. 13,219 clips total, from eval/subset.tsv."),
        ("Datasets NOT evaluated",
         "data/voxlingua107 is empty - nothing was downloaded, so no language-ID "
         "task exists. data/sib200 and data/flores_plus are TEXT only "
         "(.tsv/.jsonl); they cannot be used by these four audio-only encoders "
         "and would need a joint audio-text model such as CLAP."),
        ("One task per dataset",
         "MAEB treats each dataset as its own task, and every task here is built "
         "INSIDE a single corpus. The sole exception is Corpus_Classification, "
         "whose label IS the corpus."),
        ("", ""),
        ("Evaluators (paper section 2.2)", ""),
        ("Classification",
         "Logistic regression on the embeddings, few-shot at 8 examples per class, "
         "averaged over 10 random draws. Metric: accuracy."),
        ("Clustering",
         "MiniBatchKMeans with k set to the number of true labels. Metric: "
         "V-measure, reported here with ARI and a random baseline."),
        ("Pair Classification",
         "Cosine similarity of an audio pair, scored against the same/different "
         "label. Metric: average precision (max_ap)."),
        ("Retrieval",
         "Rank the audio corpus by cosine similarity to an audio query. Metric: "
         "CV Recall@5 - a query hits if any relevant item is in the top 5."),
        ("Reranking",
         "Rank a pre-selected candidate set of positives and hard negatives. "
         "Metric: MAP@1000."),
        ("Ranking",
         "Borda count over tasks, as in MAEB/MMTEB, reported next to the mean."),
        ("", ""),
        ("Task design", ""),
        ("Why tasks are paired",
         "The subset carries two label structures: speaker_id (WHO is speaking, "
         "acoustic) and content_group (clips sharing a transcript read by "
         "DIFFERENT speakers, linguistic). Each evaluator is instantiated on both "
         "so the paper's finding (c) - that acoustic and linguistic ability trade "
         "off - can be tested within ONE language on identical audio."),
        ("Same-speaker exclusion",
         "In every content task, same-speaker candidates are removed from the "
         "ranking. Without it a model scores well purely by matching the voice, "
         "since a speaker's own repeats sit closest. ContentReranking goes "
         "further: the query speaker's own other clips ARE the hard negatives."),
        ("", ""),
        ("Errors found and corrected", ""),
        ("Clustering sampling",
         "An earlier run subsampled individual points, which shattered the small "
         "content groups to ~1.45 members each. V-measure is not chance-corrected, "
         "so that scored 0.92 while a RANDOM partition scored 0.9195 - the metric "
         "was measuring nothing. Those numbers were discarded. Clustering now "
         "resamples whole label classes and every score ships with its baseline."),
        ("Corpus pooling confound",
         "An earlier run pooled both corpora into one index space. Because SLR30 "
         "is 48 kHz studio audio and SLR52 is 16 kHz, a 'different speaker' pair "
         "drawn across corpora is separable by recording channel alone. That "
         "inflated every speaker task: whisper speaker retrieval read 0.3595 "
         "pooled vs 0.2640 within SLR52. Correcting this changed which model "
         "ranks first - whisper was 1st pooled, w2vbert2 is 1st corrected."),
        ("", ""),
        ("Remaining limitations", ""),
        ("Content Clustering is not trustworthy",
         "Even after the fix, 3 of 4 models score at or below the random baseline "
         "and ARI is <=0.017 for all but whisper. Content groups cap at 11 members "
         "and are mostly pairs, so the many-members-per-class regime MAEB's "
         "clustering tasks assume cannot be built here. Retrieval and Reranking "
         "are the sound tests of linguistic content."),
        ("OpenSLR30 has no content tasks",
         "SLR30 has only 26 clips across 13 cross-speaker transcript groups, none "
         "with 3+ members, so its content tasks are skipped rather than reported "
         "on noise. All content results come from OpenSLR52."),
        ("Missing task types",
         "MAEB's zero-shot classification and cross-modal (text-audio) retrieval "
         "need a joint audio-text space. All four encoders here are audio-only, so "
         "those are absent - the paper prints '-' in the same columns for AST and "
         "whisper."),
        ("Final layer only",
         "Per MAEB's protocol. For the wav2vec2 family the last layer is "
         "re-specialised toward the pretraining objective and speaker information "
         "peaks mid-stack, so xlsr300m and mms300m are likely understated. "
         "Run 'python eval/maeb_eval.py --layer all' to sweep every layer."),
        ("", ""),
        ("Reproducing", ""),
        ("Commands",
         "python eval/maeb_eval.py            (writes eval/maeb_results.tsv) then "
         "python eval/build_maeb_excel.py     (writes this workbook). Embeddings "
         "come from eval/extract.py and live in eval/emb/*.npz."),
    ]
    r = 1
    for k, v in notes:
        ca = m.cell(row=r, column=1, value=k)
        cb = m.cell(row=r, column=2, value=v)
        ca.font = TITLE if v == "" else BOLD
        cb.font = BODY
        cb.alignment = Alignment(wrap_text=True, vertical="top")
        ca.alignment = Alignment(vertical="top")
        m.row_dimensions[r].height = max(14, 13 * (1 + len(v) // 100))
        r += 1
    legend(m, r, [("General terms", GENERAL_LEG), ("Metrics", METRIC_LEG)])

    wb.save(a.out)
    print(f"wrote {a.out}")
    for name in wb.sheetnames:
        print(f"  {name:24s} {wb[name].max_row:4d} rows")


if __name__ == "__main__":
    main()
