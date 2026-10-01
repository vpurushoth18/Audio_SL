#!/usr/bin/env python3
"""Build the WorldSpeech Sinhala datasets.

WorldSpeech si_lk is found speech: ~21k segments cut from real recordings, with
a `source` naming the recording, human/ASR transcripts, and quality measures.
Two datasets come out of it:

  worldspeech_sin_source   clustering by source recording. Sources with too few
                           segments are dropped and the rest are capped, so the
                           partition is not dominated by one long recording.

  worldspeech_sin_quality  classification into DNSMOS quality bands, with
                           train/test splits. Bands are cut at TERCILES of the
                           observed distribution rather than at fixed values,
                           so the classes are balanced by construction and the
                           thresholds are reported.

Audio is stored decoded to match the "array" encoding used elsewhere.
"""

import argparse
import glob
import io
import os
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SR = 16000
REPO, CONFIG = "disco-eth/WorldSpeech", "si_lk"


def audio_feature():
    return {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")}


def to_array(audio_struct, max_s):
    """Decode the stored bytes ourselves.

    WorldSpeech ships Ogg/Opus in an {bytes, path} struct. `datasets` would
    route that through torchcodec, which needs FFmpeg shared libraries this
    cluster does not have; libsndfile reads Opus directly, so soundfile does
    the job with no extra dependency.
    """
    x, sr = sf.read(io.BytesIO(audio_struct["bytes"]), dtype="float32",
                    always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x[: int(max_s * SR)]


def read_rows(split):
    """Read the cached parquet shards directly, skipping datasets' decoding."""
    base = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--disco-eth--WorldSpeech")
    files = sorted(glob.glob(
        os.path.join(base, "snapshots", "*", "data", CONFIG, f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no cached parquet for {CONFIG}/{split} under {base}. "
            f"Run: python -c \"from datasets import load_dataset; "
            f"load_dataset('{REPO}','{CONFIG}')\"")
    cols = ["audio", "source", "session_date", "dnsmos_ovr", "duration",
            "segment_id"]
    rows = []
    for f in files:
        t = pq.read_table(f, columns=cols)
        rows.extend(t.to_pylist())
    return rows


def build_source_clustering(rows, per_source, min_segments, seed, max_s):
    """Cluster by recording SESSION.

    Every Sinhala segment in WorldSpeech comes from one source
    (`sri_lanka_parliament`) with one URL, so `source` has a single value and
    cannot define clusters. `session_date` is the usable grouping: one
    parliamentary sitting, sharing acoustics and a speaker pool.
    """
    by_source = {}
    for i, r in enumerate(rows):
        by_source.setdefault(r["session_date"], []).append(i)

    rng = np.random.default_rng(seed)
    keep, sources = [], []
    for src, idx in sorted(by_source.items()):
        if len(idx) < min_segments:
            continue
        sel = idx if len(idx) <= per_source else rng.choice(
            idx, size=per_source, replace=False).tolist()
        keep.extend(sel)
        sources.append(src)

    if len(sources) < 2:
        raise ValueError(
            f"only {len(sources)} sessions have >= {min_segments} segments; "
            "lower --min-segments")

    rng.shuffle(keep)
    print(f"  {len(keep):,} segments from {len(sources)} sessions "
          f"(>= {min_segments} segments each)")

    label = ClassLabel(names=sorted(sources))
    features = Features({
        "audio": audio_feature(),
        "session_id": label,
        "segment_id": Value("string"),
    })
    ds = Dataset.from_dict(
        {
            "audio": [
                {"array": to_array(rows[i]["audio"], max_s).tolist(),
                 "sampling_rate": SR}
                for i in keep
            ],
            "session_id": [rows[i]["session_date"] for i in keep],
            "segment_id": [str(rows[i].get("segment_id", i)) for i in keep],
        },
        features=features,
    )
    return DatasetDict({"test": ds})


def build_quality(rows, n_total, seed, max_s, test_frac=0.3):
    scored = [(i, r.get("dnsmos_ovr")) for i, r in enumerate(rows)]
    scored = [(i, s) for i, s in scored if s is not None and np.isfinite(s)]
    if len(scored) < 30:
        raise ValueError("not enough segments carry a DNSMOS score")

    vals = np.array([s for _, s in scored], dtype=float)
    lo, hi = np.quantile(vals, [1 / 3, 2 / 3])
    print(f"  DNSMOS terciles: low < {lo:.3f} <= medium < {hi:.3f} <= high")

    def band(s):
        return "low" if s < lo else ("medium" if s < hi else "high")

    rng = np.random.default_rng(seed)
    by_band = {"low": [], "medium": [], "high": []}
    for i, s in scored:
        by_band[band(s)].append(i)

    per_band = max(1, n_total // 3)
    keep = []
    for b, idx in by_band.items():
        sel = idx if len(idx) <= per_band else rng.choice(
            idx, size=per_band, replace=False).tolist()
        keep.extend(sel)
        print(f"    {b:6s}: {len(sel):,} segments")

    rng.shuffle(keep)
    n_test = int(round(len(keep) * test_frac))
    splits = {"test": keep[:n_test], "train": keep[n_test:]}

    label = ClassLabel(names=["low", "medium", "high"])
    features = Features({
        "audio": audio_feature(),
        "quality_band": label,
        "segment_id": Value("string"),
    })
    out = {}
    for name, idx in splits.items():
        counts = Counter(band(dict(scored)[i]) for i in idx)
        print(f"    {name}: {len(idx):,} ({dict(counts)})")
        out[name] = Dataset.from_dict(
            {
                "audio": [
                    {"array": to_array(rows[i]["audio"], max_s).tolist(),
                     "sampling_rate": SR}
                    for i in idx
                ],
                "quality_band": [band(dict(scored)[i]) for i in idx],
                "segment_id": [str(rows[i].get("segment_id", i)) for i in idx],
            },
            features=features,
        )
    return DatasetDict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "simteb_audio/data_run"))
    ap.add_argument("--split", default="test",
                    help="WorldSpeech split to draw from (test is ~1k rows)")
    ap.add_argument("--per-source", type=int, default=20)
    ap.add_argument("--min-segments", type=int, default=5)
    ap.add_argument("--quality-total", type=int, default=1500)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", nargs="*", default=[], choices=["source", "quality"])
    a = ap.parse_args()

    print(f"reading cached parquet for {REPO} [{CONFIG}] split={a.split} ...")
    rows = read_rows(a.split)
    print(f"  {len(rows):,} segments | source={set(r['source'] for r in rows)} "
          f"| {len(set(r['session_date'] for r in rows))} sessions")

    if "source" not in a.skip:
        print("\n=== source clustering")
        dsd = build_source_clustering(rows, a.per_source, a.min_segments,
                                      a.seed, a.max_s)
        assert len(set(dsd["test"]["session_id"])) > 1
        p = os.path.join(a.out, "worldspeech_sin_session")
        dsd.save_to_disk(p)
        print(f"  saved {p}")

    if "quality" not in a.skip:
        print("\n=== quality classification")
        dsd = build_quality(rows, a.quality_total, a.seed, a.max_s)
        assert set(dsd) == {"train", "test"}
        p = os.path.join(a.out, "worldspeech_sin_quality")
        dsd.save_to_disk(p)
        print(f"  saved {p}")

    print(f"\nexport SIMTEB_AUDIO_LOCAL={a.out}")


if __name__ == "__main__":
    main()
