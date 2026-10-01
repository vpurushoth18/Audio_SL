#!/usr/bin/env python3
"""Build a unified manifest over the Sinhala speech corpora.

One row per audio file:
  dataset, utt_id, path, speaker_id, duration_s, sample_rate, channels,
  n_samples, transcript, has_transcript, n_chars
Audio duration comes from the container header (no decode), so this is I/O bound
rather than CPU bound and parallelises well over the file list.
"""
import argparse, csv, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DATA = os.path.join(ROOT, "data")

FIELDS = ["dataset", "utt_id", "path", "speaker_id", "duration_s", "sample_rate",
          "channels", "n_samples", "transcript", "has_transcript", "n_chars"]


def probe(path):
    """Header-only read: (duration_s, sample_rate, channels, n_samples) or None."""
    try:
        i = sf.info(path)
        return round(i.frames / i.samplerate, 4), i.samplerate, i.channels, i.frames
    except Exception as e:
        print(f"  !! unreadable {path}: {e}", file=sys.stderr)
        return None


def collect(entries, workers):
    """entries: list of (dataset, utt_id, path, speaker, transcript). Returns rows."""
    rows, bad = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (ds, utt, path, spk, text), info in zip(
                entries, ex.map(lambda e: probe(e[2]), entries)):
            if info is None:
                bad += 1
                continue
            dur, sr, ch, n = info
            rows.append({
                "dataset": ds, "utt_id": utt, "path": os.path.relpath(path, ROOT),
                "speaker_id": spk, "duration_s": dur, "sample_rate": sr,
                "channels": ch, "n_samples": n,
                "transcript": text or "", "has_transcript": int(bool(text)),
                "n_chars": len(text or ""),
            })
    return rows, bad


def openslr30():
    """si_lk.lines.txt lines look like:  ( sin_2241_0329430812 " text " )"""
    base = os.path.join(DATA, "openslr30", "extracted")
    if not os.path.isdir(base):
        return []
    lines = os.path.join(DATA, "openslr30", "si_lk.lines.txt")
    text = {}
    pat = re.compile(r'^\(\s*(\S+)\s+"\s*(.*?)\s*"\s*\)\s*$')
    with open(lines, encoding="utf-8") as fh:
        for ln in fh:
            m = pat.match(ln.rstrip("\n"))
            if m:
                text[m.group(1)] = m.group(2)
    out = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".wav"):
            continue
        utt = fn[:-4]
        parts = utt.split("_")          # sin_<speaker>_<uttnum>
        spk = parts[1] if len(parts) > 2 else ""
        out.append(("openslr30", utt, os.path.join(base, fn), spk, text.get(utt, "")))
    return out


def openslr52():
    """utt_spk_text.tsv: <utt_id>\t<speaker_id>\t<transcript>; audio in data/<xx>/<utt>.flac"""
    base = os.path.join(DATA, "openslr52", "extracted", "asr_sinhala")
    if not os.path.isdir(base):
        return []
    meta = {}
    tsv = os.path.join(base, "utt_spk_text.tsv")
    if not os.path.exists(tsv):
        tsv = os.path.join(DATA, "openslr52", "utt_spk_text.tsv")
    with open(tsv, encoding="utf-8") as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                meta[p[0]] = (p[1], p[2])
    out = []
    adir = os.path.join(base, "data")
    for sub in sorted(os.listdir(adir)):
        sd = os.path.join(adir, sub)
        if not os.path.isdir(sd):
            continue
        for fn in sorted(os.listdir(sd)):
            if not fn.endswith(".flac"):
                continue
            utt = fn[:-5]
            spk, txt = meta.get(utt, ("", ""))
            out.append(("openslr52", utt, os.path.join(sd, fn), spk, txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "manifest.tsv"))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    all_rows = []
    for name, fn in (("openslr30", openslr30), ("openslr52", openslr52)):
        entries = fn()
        if not entries:
            print(f"[{name}] not extracted yet - skipping")
            continue
        print(f"[{name}] probing {len(entries):,} files ...")
        rows, bad = collect(entries, args.workers)
        tot = sum(r["duration_s"] for r in rows)
        withtxt = sum(r["has_transcript"] for r in rows)
        print(f"[{name}] {len(rows):,} ok, {bad} unreadable | {tot/3600:.2f} h | "
              f"{withtxt:,} with transcript ({withtxt/max(len(rows),1)*100:.1f}%) | "
              f"{len({r['speaker_id'] for r in rows}):,} speakers")
        all_rows.extend(rows)

    if not all_rows:
        print("nothing to write"); return

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
