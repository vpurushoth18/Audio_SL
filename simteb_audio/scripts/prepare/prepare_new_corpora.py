#!/usr/bin/env python3
"""Build the Omnilingual and VoxLingua107 Sinhala datasets.

Both are read straight from their cached downloads and decoded with soundfile,
not through `datasets`' Audio feature, which needs torchcodec + FFmpeg (absent
on this cluster). Audio is stored as decoded float32 arrays, matching the
"array" encoding the other prepared datasets use.

Produces, under --out:
  omnilingual_sin_speaker      speaker clustering   (test)
  omnilingual_sin_speaker_clf  speaker classification (train/test)
  omnilingual_sin_retrieval    cross-speaker content retrieval (test)
  voxlingua_sin_video          video clustering     (test)

Omnilingual has only 7 Sinhala speakers, so the speaker tasks are much easier
than the OpenSLR-52 ones; that is documented on the tasks themselves.
"""

import argparse
import collections
import io
import os
import tarfile

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from datasets import (ClassLabel, Dataset, DatasetDict, Features, Sequence,
                      Value)

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SR = 16000

OMNI_REPO = "facebook/omnilingual-asr-corpus"
OMNI_FILES = [
    "data/sin_Sinh/train-00000-of-00002.parquet",
    "data/sin_Sinh/train-00001-of-00002.parquet",
    "data/sin_Sinh/test-00000-of-00001.parquet",
    "data/sin_Sinh/dev-00000-of-00001.parquet",
]
VOX_REPO = "TalTechNLP/voxlingua107_wds"


def decode(raw, max_s):
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x[: int(max_s * SR)]


def omni_uid(r):
    """Omnilingual's `segment_id` is the constant "s01" for every Sinhala row,
    so it cannot identify an utterance. (speaker, prompt, segment) is unique."""
    return f"{r['speaker_id']}__{r['prompt_id']}__{r['segment_id']}"


def audio_feature():
    return {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")}


def arrow_audio(rows, key, max_s):
    return [{"array": decode(r[key]["bytes"], max_s).tolist(),
             "sampling_rate": SR} for r in rows]


# --------------------------------------------------------------- omnilingual
def load_omni():
    from huggingface_hub import hf_hub_download

    cols = ["audio", "speaker_id", "prompt_id", "segment_id", "duration"]
    rows = []
    for f in OMNI_FILES:
        p = hf_hub_download(OMNI_REPO, f, repo_type="dataset")
        rows.extend(pq.read_table(p, columns=cols).to_pylist())
    return rows


def omni_speaker(rows, max_s, per_speaker, seed):
    rng = np.random.default_rng(seed)
    by_spk = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_spk[r["speaker_id"]].append(i)

    keep = []
    for spk, idx in sorted(by_spk.items()):
        sel = idx if len(idx) <= per_speaker else rng.choice(
            idx, size=per_speaker, replace=False).tolist()
        keep.extend(sel)
    rng.shuffle(keep)
    speakers = sorted({rows[i]["speaker_id"] for i in keep})
    print(f"  clustering: {len(keep):,} utts, {len(speakers)} speakers")

    feats = Features({
        "audio": audio_feature(),
        "speaker_id": ClassLabel(names=speakers),
        "utt_id": Value("string"),
    })
    sub = [rows[i] for i in keep]
    ds = Dataset.from_dict({
        "audio": arrow_audio(sub, "audio", max_s),
        "speaker_id": [r["speaker_id"] for r in sub],
        "utt_id": [omni_uid(r) for r in sub],
    }, features=feats)
    return DatasetDict({"test": ds}), speakers


def omni_speaker_clf(rows, max_s, per_speaker, seed, test_frac=0.3):
    rng = np.random.default_rng(seed)
    by_spk = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_spk[r["speaker_id"]].append(i)

    train_idx, test_idx = [], []
    for spk, idx in sorted(by_spk.items()):
        idx = list(idx)
        rng.shuffle(idx)
        if len(idx) > per_speaker:
            idx = idx[:per_speaker]
        n_test = max(1, int(round(len(idx) * test_frac)))
        if len(idx) - n_test < 1:
            continue
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    speakers = sorted({rows[i]["speaker_id"] for i in train_idx + test_idx})
    feats = Features({
        "audio": audio_feature(),
        "speaker_id": ClassLabel(names=speakers),
        "utt_id": Value("string"),
    })
    out = {}
    for name, idx in (("train", train_idx), ("test", test_idx)):
        sub = [rows[i] for i in idx]
        print(f"  classification {name}: {len(sub):,} utts, "
              f"{len({r['speaker_id'] for r in sub})} speakers")
        out[name] = Dataset.from_dict({
            "audio": arrow_audio(sub, "audio", max_s),
            "speaker_id": [r["speaker_id"] for r in sub],
            "utt_id": [omni_uid(r) for r in sub],
        }, features=feats)
    return DatasetDict(out)


def omni_retrieval(rows, max_s):
    """Keep only prompts with more than one speaker - the rest can never match."""
    by_prompt = collections.defaultdict(set)
    for r in rows:
        by_prompt[r["prompt_id"]].add(r["speaker_id"])
    usable = {p for p, s in by_prompt.items() if len(s) > 1}
    sub = [r for r in rows if r["prompt_id"] in usable]
    print(f"  retrieval: {len(sub):,} utts over {len(usable)} cross-speaker prompts")
    if not sub:
        raise ValueError("no cross-speaker prompts")

    feats = Features({
        "audio": audio_feature(),
        "utt_id": Value("string"),
        "sentence_id": Value("string"),
        "speaker_id": Value("string"),
        "transcript": Value("string"),
    })
    ds = Dataset.from_dict({
        "audio": arrow_audio(sub, "audio", max_s),
        "utt_id": [omni_uid(r) for r in sub],
        "sentence_id": [str(r["prompt_id"]) for r in sub],
        "speaker_id": [str(r["speaker_id"]) for r in sub],
        "transcript": ["" for _ in sub],
    }, features=feats)
    return DatasetDict({"test": ds})


# --------------------------------------------------------------- voxlingua
def vox_video(n_shards, per_video, min_clips, max_s, seed):
    from huggingface_hub import hf_hub_download

    clips = []          # (video_id, utt_id, wav bytes)
    for i in range(n_shards):
        p = hf_hub_download(VOX_REPO, f"train/si/{i:06d}.tar", repo_type="dataset")
        with tarfile.open(p) as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".wav"):
                    continue
                utt = m.name[:-4]
                clips.append((utt.split("__U__")[0], utt,
                              tf.extractfile(m).read()))
        print(f"    shard {i}: {len(clips):,} clips")

    by_video = collections.defaultdict(list)
    for j, (vid, _, _) in enumerate(clips):
        by_video[vid].append(j)

    rng = np.random.default_rng(seed)
    keep, videos = [], []
    for vid, idx in sorted(by_video.items()):
        if len(idx) < min_clips:
            continue
        sel = idx if len(idx) <= per_video else rng.choice(
            idx, size=per_video, replace=False).tolist()
        keep.extend(sel)
        videos.append(vid)

    if len(videos) < 2:
        raise ValueError(f"only {len(videos)} videos have >= {min_clips} clips")
    rng.shuffle(keep)
    print(f"  video clustering: {len(keep):,} clips from {len(videos)} videos "
          f"(>= {min_clips} clips each)")

    feats = Features({
        "audio": audio_feature(),
        "video_id": ClassLabel(names=sorted(videos)),
        "utt_id": Value("string"),
    })
    ds = Dataset.from_dict({
        "audio": [{"array": decode(clips[j][2], max_s).tolist(),
                   "sampling_rate": SR} for j in keep],
        "video_id": [clips[j][0] for j in keep],
        "utt_id": [clips[j][1] for j in keep],
    }, features=feats)
    return DatasetDict({"test": ds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "simteb_audio/data_run"))
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--omni-per-speaker", type=int, default=60)
    ap.add_argument("--vox-shards", type=int, default=6)
    ap.add_argument("--vox-per-video", type=int, default=10)
    ap.add_argument("--vox-min-clips", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=[],
                    choices=["omni", "vox"])
    a = ap.parse_args()

    want = a.only or ["omni", "vox"]

    if "omni" in want:
        print("=== Omnilingual ASR, sin_Sinh")
        rows = load_omni()
        print(f"  {len(rows):,} utterances, "
              f"{len({r['speaker_id'] for r in rows})} speakers")

        dsd, speakers = omni_speaker(rows, a.max_s, a.omni_per_speaker, a.seed)
        assert len(set(dsd["test"]["speaker_id"])) > 1
        p = os.path.join(a.out, "omnilingual_sin_speaker")
        dsd.save_to_disk(p); print(f"    saved {p}")

        dsd = omni_speaker_clf(rows, a.max_s, a.omni_per_speaker, a.seed)
        assert set(dsd) == {"train", "test"}
        p = os.path.join(a.out, "omnilingual_sin_speaker_clf")
        dsd.save_to_disk(p); print(f"    saved {p}")

        dsd = omni_retrieval(rows, a.max_s)
        assert len(set(dsd["test"]["utt_id"])) == len(dsd["test"])
        p = os.path.join(a.out, "omnilingual_sin_retrieval")
        dsd.save_to_disk(p); print(f"    saved {p}")

    if "vox" in want:
        print("\n=== VoxLingua107, si")
        dsd = vox_video(a.vox_shards, a.vox_per_video, a.vox_min_clips,
                        a.max_s, a.seed)
        assert len(set(dsd["test"]["video_id"])) > 1
        p = os.path.join(a.out, "voxlingua_sin_video")
        dsd.save_to_disk(p); print(f"    saved {p}")

    print(f"\nexport SIMTEB_AUDIO_LOCAL={a.out}")


if __name__ == "__main__":
    main()
