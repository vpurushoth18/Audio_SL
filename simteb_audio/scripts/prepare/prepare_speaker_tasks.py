#!/usr/bin/env python3
"""Build the classification and pair-classification datasets.

Both derive from the same per-speaker subsample the clustering datasets use, but
need different shapes:

  classification  train + test splits, stratified by speaker, so MTEB can fit a
                  probe on train and score accuracy on test. Speakers with too
                  few utterances to appear in both splits are dropped.

  pairs           one row per PAIR, with audio1/audio2 and a 0/1 label.
                  Positives are two utterances of one speaker; negatives are two
                  speakers. Balanced 50/50, and no utterance pairs with itself.

Audio is stored as decoded arrays so it runs without FFmpeg, matching the
"array" encoding used elsewhere.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from datasets import (ClassLabel, Dataset, DatasetDict, Features, Sequence,
                      Value)

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SR = 16000

SPECS = {
    "openslr52": dict(clf="openslr52_speaker_clf", pairs="openslr52_speaker_pairs"),
    "openslr30": dict(clf="openslr30_speaker_clf", pairs="openslr30_speaker_pairs"),
}


def load_audio(path, target_sr=SR, max_s=20.0):
    x, sr = sf.read(os.path.join(ROOT, path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != target_sr:
        x = soxr.resample(x, sr, target_sr)
    return x[: int(max_s * target_sr)]


def subsample(manifest, name, per_speaker, min_utts, min_dur, max_dur, seed):
    d = manifest[manifest.dataset == name].copy()
    d = d[(d.duration_s >= min_dur) & (d.duration_s <= max_dur)]
    counts = d.speaker_id.value_counts()
    d = d[d.speaker_id.isin(counts[counts >= min_utts].index)]
    keep = []
    for _, g in d.groupby("speaker_id", sort=True):
        keep.extend(g.sample(n=min(len(g), per_speaker),
                             random_state=seed).index.tolist())
    d = d.loc[keep].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return d


def audio_feature():
    return {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")}


def build_classification(d, seed, test_frac=0.3):
    """Stratified per-speaker split so every speaker appears in both splits."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, g in d.groupby("speaker_id", sort=True):
        idx = g.index.to_numpy().copy()   # to_numpy may be read-only
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        # a speaker needs at least one utterance on each side to be usable
        if len(idx) - n_test < 1:
            continue
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    speakers = sorted(d.loc[train_idx + test_idx].speaker_id.unique())
    label = ClassLabel(names=[str(s) for s in speakers])
    features = Features({
        "audio": audio_feature(),
        "speaker_id": label,
        "utt_id": Value("string"),
    })

    out = {}
    for split, idx in (("train", train_idx), ("test", test_idx)):
        sub = d.loc[idx]
        print(f"    {split}: {len(sub):,} utts, {sub.speaker_id.nunique()} speakers")
        out[split] = Dataset.from_dict(
            {
                "audio": [
                    {"array": load_audio(p).tolist(), "sampling_rate": SR}
                    for p in sub.path
                ],
                "speaker_id": [str(s) for s in sub.speaker_id],
                "utt_id": sub.utt_id.tolist(),
            },
            features=features,
        )
    return DatasetDict(out), len(speakers)


def build_pairs(d, seed, n_pairs):
    """Balanced same-speaker / different-speaker pairs."""
    rng = np.random.default_rng(seed)
    by_spk = {s: g.index.to_numpy() for s, g in d.groupby("speaker_id", sort=True)}
    usable = [s for s, idx in by_spk.items() if len(idx) >= 2]
    if len(usable) < 2:
        raise ValueError("need at least two speakers with two utterances each")

    half = n_pairs // 2
    pos, neg = set(), set()

    while len(pos) < half:
        s = usable[rng.integers(len(usable))]
        i, j = rng.choice(by_spk[s], size=2, replace=False)
        pos.add((int(min(i, j)), int(max(i, j))))

    while len(neg) < half:
        a, b = rng.choice(len(usable), size=2, replace=False)
        i = int(rng.choice(by_spk[usable[a]]))
        j = int(rng.choice(by_spk[usable[b]]))
        neg.add((min(i, j), max(i, j)))

    pairs = [(i, j, 1) for i, j in pos] + [(i, j, 0) for i, j in neg]
    rng.shuffle(pairs)
    print(f"    {len(pairs):,} pairs ({sum(p[2] for p in pairs):,} positive)")

    # decode each needed utterance once, not once per pair it appears in
    needed = sorted({i for p in pairs for i in p[:2]})
    cache = {i: load_audio(d.loc[i, "path"]).tolist() for i in needed}

    features = Features({
        "audio1": audio_feature(),
        "audio2": audio_feature(),
        "labels": Value("int64"),
    })
    ds = Dataset.from_dict(
        {
            "audio1": [{"array": cache[i], "sampling_rate": SR} for i, _, _ in pairs],
            "audio2": [{"array": cache[j], "sampling_rate": SR} for _, j, _ in pairs],
            "labels": [lab for _, _, lab in pairs],
        },
        features=features,
    )
    return DatasetDict({"test": ds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/manifest.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "simteb_audio/data_run"))
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--min-utts", type=int, default=10)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--pairs", type=int, default=4000,
                    help="total pairs for the verification task")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", nargs="*", default=[], choices=["clf", "pairs"])
    a = ap.parse_args()

    manifest = pd.read_csv(a.manifest, sep="\t", quoting=3,
                           dtype={"speaker_id": str})

    for name in a.datasets:
        spec = SPECS[name]
        print(f"\n=== {name}")
        d = subsample(manifest, name, a.per_speaker, a.min_utts,
                      a.min_dur, a.max_dur, a.seed)
        print(f"  pool: {len(d):,} utts, {d.speaker_id.nunique()} speakers")

        if "clf" not in a.skip:
            print("  building classification splits")
            dsd, n_spk = build_classification(d, a.seed)
            assert set(dsd) == {"train", "test"}
            assert n_spk > 1
            p = os.path.join(a.out, spec["clf"])
            dsd.save_to_disk(p)
            print(f"    saved {p}")

        if "pairs" not in a.skip:
            print("  building verification pairs")
            n = min(a.pairs, len(d) * 4)
            dsd = build_pairs(d, a.seed, n)
            counts = Counter(dsd["test"]["labels"])
            assert counts[0] > 0 and counts[1] > 0, counts
            p = os.path.join(a.out, spec["pairs"])
            dsd.save_to_disk(p)
            print(f"    saved {p}  (labels: {dict(counts)})")

    print(f"\nexport SIMTEB_AUDIO_LOCAL={a.out}")


if __name__ == "__main__":
    main()
