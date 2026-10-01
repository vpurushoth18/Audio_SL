#!/usr/bin/env python3
"""Build the shared dataset inventory workbook.

One sheet, one row per dataset, meant to be edited by hand as new datasets are
added. Figures for the local datasets are measured from data/manifest.tsv and
the files on disk, not copied from dataset cards.
"""
import argparse, os
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
FONT = "Arial"
BODY = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=11, bold=True)
GREY = Font(name=FONT, size=10, color="808080")
UNDER = Border(bottom=Side(style="thin", color="000000"))
BLANK_FILL = PatternFill("solid", fgColor="FFF9E6")

COLS = [
    ("Dataset", 20), ("Source / ID", 30), ("Language", 12), ("Modality", 14),
    ("Audio files", 11), ("Items", 10), ("Hours", 9), ("Speakers", 10),
    ("Sample rate", 12), ("Transcripts", 12), ("Size on disk", 13),
    ("Local path", 34), ("Status", 24), ("Used for", 24), ("Notes", 46),
]

ROWS = [
    ["OpenSLR-52", "openslr.org/52 (asr_sinhala)", "Sinhala", "Speech + text",
     "Yes", 185293, 224.50, 478, "16 kHz", "100%", "28 GB",
     "data/openslr52/extracted", "Downloaded and processed",
     "Embedding evaluation",
     "Large Sinhala ASR corpus. 185,293 FLAC files. Every utterance has a "
     "transcript. Main dataset for the embedding work so far."],

    ["OpenSLR-30", "openslr.org/30 (si_lk)", "Sinhala", "Speech + text",
     "Yes", 2064, 3.38, 12, "48 kHz", "61%", "1.8 GB",
     "data/openslr30/extracted", "Downloaded and processed",
     "Embedding evaluation",
     "Small high-quality multi-speaker set. Only 61% of utterances have a "
     "transcript. Used as the clean 48 kHz contrast; resampled to 16 kHz."],

    ["VoxLingua107", "bark.phon.ioc.ee/voxlingua107", "Sinhala subset",
     "Speech", "No", None, None, None, "16 kHz", "No", "0 B",
     "data/voxlingua107 (empty)", "Not downloaded",
     "Language ID / pretraining",
     "Directory created but nothing fetched yet. Audio is available upstream "
     "but is not on disk, so no figures can be given."],

    ["FLORES+", "openlanguagedata/flores_plus", "Sinhala (sin_Sinh)", "Text",
     "No", 2009, None, None, "n/a", "n/a", "1.3 MB",
     "data/flores_plus", "Downloaded", "Translation / text evaluation",
     "No audio. 997 dev + 1,012 devtest sentences, as JSONL."],

    ["SIB-200", "Davlan/sib200", "Sinhala", "Text",
     "No", 1007, None, None, "n/a", "n/a", "364 KB",
     "data/sib200", "Downloaded", "Topic classification",
     "No audio. 702 train / 100 dev / 205 test, labelled over 7 topics "
     "(science-technology, travel, politics, sports, health, entertainment, "
     "geography)."],

    ["LaMuN", "tharindu/LaMuN (sin split)", "Sinhala", "Text + image",
     "No", 3416, None, None, "n/a", "n/a", "4.8 GB (all languages)",
     "HF cache: datasets--tharindu--LaMuN", "Downloaded",
     "Not yet assigned",
     "No audio. Sinhala split only: 2,416 train + 1,000 test news items with "
     "image, caption, title, content and source. Other languages (ara, ind, "
     "...) are in the same download."],
]

LEGEND = [
    ("How to use this sheet", ""),
    ("Editing", "Add one row per dataset. The shaded rows at the bottom are blank "
                "templates - overwrite them and add more as needed."),
    ("Audio files", "Yes or No. 'No' means the dataset has no speech audio at all, "
                    "or the audio exists upstream but is not on our disk - the "
                    "Status column says which."),
    ("Items", "Utterances for speech datasets, sentences or documents for text ones."),
    ("Hours", "Total audio duration. Blank where there is no audio."),
    ("Transcripts", "Share of utterances that carry a transcript."),
    ("Status", "Not downloaded / Downloaded / Downloaded and processed."),
    ("", ""),
    ("Where the numbers come from", ""),
    ("Measured", "Items, Hours, Speakers, Sample rate and Transcripts for OpenSLR-52 "
                 "and OpenSLR-30 are measured from data/manifest.tsv, not taken from "
                 "the dataset cards."),
    ("Sizes", "Size on disk is what the files occupy here, including the original "
              "archives where those were kept."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/sinhala_datasets.xlsx"))
    ap.add_argument("--blank-rows", type=int, default=6)
    a = ap.parse_args()

    wb = Workbook()
    ws = wb.active
    ws.title = "Datasets"

    for i, (_, w) in enumerate(COLS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.cell(row=1, column=1, value="Sinhala datasets").font = TITLE

    hr = 3
    for i, (name, _) in enumerate(COLS, start=1):
        c = ws.cell(row=hr, column=i, value=name)
        c.font, c.border = BOLD, UNDER
        c.alignment = Alignment(vertical="bottom", wrap_text=True)

    r = hr + 1
    for row in ROWS:
        for i, v in enumerate(row, start=1):
            c = ws.cell(row=r, column=i, value=v)
            c.font = BODY
            c.alignment = Alignment(vertical="top",
                                    wrap_text=(COLS[i - 1][0] == "Notes"))
            if COLS[i - 1][0] == "Hours" and v is not None:
                c.number_format = "0.00"
            if COLS[i - 1][0] in ("Items", "Speakers") and v is not None:
                c.number_format = "#,##0"
        ws.row_dimensions[r].height = 42
        r += 1

    first_data, last_data = hr + 1, r - 1

    # blank rows to fill in
    for _ in range(a.blank_rows):
        for i in range(1, len(COLS) + 1):
            c = ws.cell(row=r, column=i)
            c.fill, c.font, c.border = BLANK_FILL, BODY, Border(
                bottom=Side(style="hair", color="D0D0D0"))
        r += 1
    last_blank = r - 1

    # totals over the audio datasets
    r += 1
    ws.cell(row=r, column=1, value="Total").font = BOLD
    tc = ws.cell(row=r, column=6,
                 value=f"=SUM(F{first_data}:F{last_blank})")
    tc.font, tc.number_format = BOLD, "#,##0"
    th = ws.cell(row=r, column=7,
                 value=f"=SUM(G{first_data}:G{last_blank})")
    th.font, th.number_format = BOLD, "0.00"
    ws.cell(row=r, column=15,
            value="Hours count audio datasets only.").font = GREY

    # dropdowns
    dv_audio = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    dv_status = DataValidation(
        type="list",
        formula1='"Not downloaded,Downloaded,Downloaded and processed"',
        allow_blank=True)
    ws.add_data_validation(dv_audio)
    ws.add_data_validation(dv_status)
    dv_audio.add(f"E{first_data}:E{last_blank}")
    dv_status.add(f"M{first_data}:M{last_blank}")

    # legend, below the table
    lr = last_blank + 4
    for k, v in LEGEND:
        ws.cell(row=lr, column=1, value=k).font = BOLD if v == "" else BODY
        c = ws.cell(row=lr, column=2, value=v)
        c.font = BODY
        c.alignment = Alignment(vertical="top")
        lr += 1

    ws.freeze_panes = f"A{hr+1}"
    ws.auto_filter.ref = f"A{hr}:{get_column_letter(len(COLS))}{last_blank}"

    wb.save(a.out)
    print(f"wrote {a.out}")
    print(f"  {len(ROWS)} datasets, {a.blank_rows} blank rows")


if __name__ == "__main__":
    main()
