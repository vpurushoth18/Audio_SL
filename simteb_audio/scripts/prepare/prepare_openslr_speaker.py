#!/usr/bin/env python3
"""Build the speaker-clustering datasets from data/manifest.tsv.

Writes a DatasetDict with a single "test" split holding an Audio column and an
integer speaker_id, saves it to disk, and optionally pushes it to the Hub.

Clustering needs a balanced subsample rather than the whole corpus: OpenSLR-52
has 185,293 utterances, and speakers with wildly different utterance counts make
V-measure reflect the imbalance more than the embeddings. Each speaker is capped
at --per-speaker utterances and speakers below --min-utts are dropped.

By default nothing is uploaded. Pass --push to publish.
"""

import argparse
import os
from collections import Counter

import pandas as pd
import numpy as np
import soundfile as sf
import soxr
from datasets import (Audio, ClassLabel, Dataset, DatasetDict, Features,
                      Sequence, Value)

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"

SPECS = {
    "openslr52": dict(
        dir_name="openslr52_speaker",
        repo="Sinhala-NLP/SiMTEB-Audio-OpenSLR52-Speaker",
        sampling_rate=16000,
    ),
    "openslr30": dict(
        dir_name="openslr30_speaker",
        repo="Sinhala-NLP/SiMTEB-Audio-OpenSLR30-Speaker",
        sampling_rate=16000,
    ),
}


def _load_audio(path, target_sr):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != target_sr:
        x = soxr.resample(x, sr, target_sr)
    return x


def build(manifest, dataset_name, per_speaker, min_utts, min_dur, max_dur, seed,
          encoding):
    d = manifest[manifest.dataset == dataset_name].copy()
    if d.empty:
        raise ValueError(f"no rows for dataset {dataset_name!r} in the manifest")

    before = len(d)
    d = d[(d.duration_s >= min_dur) & (d.duration_s <= max_dur)]
    print(f"  duration filter [{min_dur}, {max_dur}]s: {before:,} -> {len(d):,}")

    counts = d.speaker_id.value_counts()
    keep = counts[counts >= min_utts].index
    dropped = len(counts) - len(keep)
    d = d[d.speaker_id.isin(keep)]
    print(f"  speakers with >= {min_utts} utts: {len(keep)} kept, {dropped} dropped")

    # cap per speaker so no one speaker dominates the partition.
    # done by collecting indices rather than groupby.apply, which drops the
    # grouping column in pandas 2.x
    keep_idx = []
    for _, g in d.groupby("speaker_id", sort=True):
        keep_idx.extend(g.sample(n=min(len(g), per_speaker),
                                 random_state=seed).index.tolist())
    d = d.loc[keep_idx]
    d = d.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    speakers = sorted(d.speaker_id.unique())
    print(f"  final: {len(d):,} utterances, {len(speakers)} speakers, "
          f"{d.duration_s.sum() / 3600:.2f} h")
    per = d.speaker_id.value_counts()
    print(f"  utts/speaker: min={per.min()} median={per.median():.0f} max={per.max()}")

    sr = SPECS[dataset_name]["sampling_rate"]
    label = ClassLabel(names=[str(s) for s in speakers])
    common = {
        "speaker_id": [str(s) for s in d.speaker_id],
        "utt_id": d.utt_id.tolist(),
        "duration_s": d.duration_s.astype("float32").tolist(),
    }

    if encoding == "audio":
        # Idiomatic HF Audio column holding the ORIGINAL encoded bytes - this is
        # what MTEB datasets on the Hub look like, and it is far smaller than
        # decoded audio. Built as a raw {bytes, path} struct and then cast,
        # because Audio.encode_example imports torchcodec unconditionally even
        # though storing bytes never needs a codec. Decoding at read time does
        # need torchcodec + FFmpeg; this machine has neither, which is why the
        # "array" encoding also exists.
        print(f"  reading {len(d):,} files as bytes...")
        blobs = []
        for p_ in d.path:
            fp = os.path.join(ROOT, p_)
            with open(fp, "rb") as fh:
                blobs.append({"bytes": fh.read(), "path": os.path.basename(fp)})
        total = sum(len(b["bytes"]) for b in blobs)
        print(f"  {total / 2**30:.2f} GiB of encoded audio "
              f"(vs {sum(d.n_samples) * 4 / 2**30:.2f} GiB decoded)")
        features = Features({
            "audio": {"bytes": Value("binary"), "path": Value("string")},
            "speaker_id": label,
            "utt_id": Value("string"),
            "duration_s": Value("float32"),
        })
        data = {"audio": blobs, **common}
        ds = Dataset.from_dict(data, features=features)
        ds = ds.cast_column("audio", Audio(sampling_rate=sr))
        return DatasetDict({"test": ds})

    # decoded arrays as plain features. Larger on disk, but needs no codec:
    # MTEB's AudioCollator only requires {"array", "sampling_rate"}
    print(f"  decoding {len(d):,} files to arrays at {sr} Hz...")
    arrays = [_load_audio(os.path.join(ROOT, p), sr) for p in d.path]
    total = sum(a.nbytes for a in arrays)
    print(f"  decoded {total / 2**30:.2f} GiB of float32 audio")
    features = Features({
        "audio": {
            "array": Sequence(Value("float32")),
            "sampling_rate": Value("int32"),
        },
        "speaker_id": label,
        "utt_id": Value("string"),
        "duration_s": Value("float32"),
    })
    data = {
        "audio": [{"array": a.tolist(), "sampling_rate": sr} for a in arrays],
        **common,
    }

    ds = Dataset.from_dict(data, features=features)
    return DatasetDict({"test": ds})


def validate(dsd, min_utts, encoding="array"):
    ds = dsd["test"]
    assert set(dsd.keys()) == {"test"}, dsd.keys()
    assert "audio" in ds.column_names and "speaker_id" in ds.column_names

    labels = ds["speaker_id"]
    counts = Counter(labels)
    n_spk = len(counts)
    assert n_spk > 1, f"clustering needs more than one speaker, found {n_spk}"
    assert min(counts.values()) >= min(min_utts, 2), \
        f"a speaker has only {min(counts.values())} utterances"

    if encoding == "array":
        # the first example must actually decode, otherwise the paths are wrong
        ex = ds[0]
        arr = ex["audio"]["array"]
        assert len(arr) > 0, "first example decoded to an empty array"
        print(f"  validated: {len(ds):,} rows, {n_spk} speakers, "
              f"first example {len(arr):,} samples @ "
              f"{ex['audio']['sampling_rate']} Hz")
    else:
        # decoding needs torchcodec, which is absent here; check the stored
        # bytes are non-empty instead
        # read the Arrow column directly; any Dataset access decodes
        blob = ds.data.column("audio")[0].as_py()
        nbytes = len(blob["bytes"]) if blob and blob.get("bytes") else 0
        assert nbytes > 0, "first example stored zero bytes"
        print(f"  validated: {len(ds):,} rows, {n_spk} speakers, "
              f"first example {nbytes:,} bytes stored (not decoded here)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SPECS),
                    choices=list(SPECS))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/manifest.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "simteb_audio/data"))
    ap.add_argument("--per-speaker", type=int, default=20,
                    help="max utterances kept per speaker")
    ap.add_argument("--min-utts", type=int, default=10,
                    help="drop speakers with fewer utterances than this")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoding", choices=["array", "audio"], default="array",
                    help="array: decoded float32, no codec needed (default). "
                         "audio: HF Audio column, needs torchcodec + FFmpeg")
    ap.add_argument("--push", action="store_true",
                    help="upload to the Hub (nothing is uploaded without this)")
    a = ap.parse_args()

    manifest = pd.read_csv(a.manifest, sep="\t", quoting=3,
                           dtype={"speaker_id": str})

    for name in a.datasets:
        spec = SPECS[name]
        print(f"\n=== {name} -> {spec['dir_name']}")
        dsd = build(manifest, name, a.per_speaker, a.min_utts,
                    a.min_dur, a.max_dur, a.seed, a.encoding)
        validate(dsd, a.min_utts, a.encoding)

        path = os.path.join(a.out, spec["dir_name"])
        os.makedirs(a.out, exist_ok=True)
        dsd.save_to_disk(path)
        print(f"  saved to {path}")

        if a.push:
            print(f"  pushing to {spec['repo']}")
            dsd.push_to_hub(spec["repo"])
            from huggingface_hub import HfApi
            info = HfApi().dataset_info(spec["repo"])
            print(f"  revision: {info.sha}   <- put this in the task metadata")

    print(f"\nTo run the tasks against these local copies:\n"
          f"  export SIMTEB_AUDIO_LOCAL={a.out}")
    if not a.push:
        print("Nothing was uploaded. Re-run with --push when you are ready.")


if __name__ == "__main__":
    main()
