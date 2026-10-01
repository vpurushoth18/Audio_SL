#!/usr/bin/env python3
"""Build the OpenSLR-52 retrieval and reranking datasets.

Retrieval needs a different sample from the clustering tasks: it must be built
around CONTENT groups (utterances sharing a transcript, read by different
speakers), not around speakers, because the correct answers are defined by what
was said rather than by who said it.

  <out>/openslr52_retrieval   utt_id, sentence_id, speaker_id, transcript, audio
                              -> A2T, T2A and A2A tasks
  <out>/openslr52_rerank      the same, plus query_id / positive_ids marking a
                              per-query candidate pool of cross-speaker
                              positives and same-speaker hard negatives

Audio is stored decoded, matching the "array" encoding used elsewhere so the
tasks run without FFmpeg.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from datasets import (Dataset, DatasetDict, Features, Sequence, Value)

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SR = 16000


def load_audio(path, max_s=20.0):
    x, sr = sf.read(os.path.join(ROOT, path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x[: int(max_s * SR)]


def audio_feature():
    return {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")}


def pick_groups(manifest, n_groups, min_speakers, min_dur, max_dur, seed):
    """Content groups with at least `min_speakers` distinct speakers."""
    d = manifest[(manifest.dataset == "openslr52") &
                 (manifest.has_transcript == 1)].copy()
    d = d[(d.duration_s >= min_dur) & (d.duration_s <= max_dur)]
    d["norm"] = (d.transcript.fillna("").astype(str)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    d = d[d.norm != ""]

    g = d.groupby("norm").agg(n=("utt_id", "size"),
                              nspk=("speaker_id", "nunique"))
    eligible = g[(g.n > 1) & (g.nspk >= min_speakers)].index.to_numpy()
    if len(eligible) == 0:
        raise ValueError("no content groups with enough distinct speakers")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_groups, len(eligible)),
                        replace=False)
    sub = d[d.norm.isin(chosen)].copy()
    sub["sentence_id"] = pd.factorize(sub.norm)[0]
    sub["sentence_id"] = "s" + sub.sentence_id.astype(str)
    print(f"  {len(chosen):,} content groups -> {len(sub):,} utterances, "
          f"{sub.speaker_id.nunique()} speakers, "
          f"{sub.duration_s.sum()/3600:.2f} h")
    per = sub.groupby("sentence_id").size()
    print(f"  utts/sentence: min={per.min()} median={per.median():.0f} "
          f"max={per.max()}")
    return sub.reset_index(drop=True)


def build_retrieval(sub):
    features = Features({
        "audio": audio_feature(),
        "utt_id": Value("string"),
        "sentence_id": Value("string"),
        "speaker_id": Value("string"),
        "transcript": Value("string"),
    })
    print(f"  decoding {len(sub):,} files...")
    ds = Dataset.from_dict(
        {
            "audio": [{"array": load_audio(p).tolist(), "sampling_rate": SR}
                      for p in sub.path],
            "utt_id": sub.utt_id.astype(str).tolist(),
            "sentence_id": sub.sentence_id.tolist(),
            "speaker_id": sub.speaker_id.astype(str).tolist(),
            "transcript": sub.transcript.astype(str).tolist(),
        },
        features=features,
    )
    return DatasetDict({"test": ds})


def build_rerank(sub, n_queries, n_negatives, seed):
    """Per query: cross-speaker positives + same-speaker hard negatives."""
    rng = np.random.default_rng(seed)
    by_sentence = defaultdict(list)
    by_speaker = defaultdict(list)
    for i, r in enumerate(sub.itertuples()):
        by_sentence[r.sentence_id].append(i)
        by_speaker[r.speaker_id].append(i)

    rows = []
    candidates = set()
    order = rng.permutation(len(sub))
    for i in order:
        if len(rows) >= n_queries:
            break
        r = sub.iloc[i]
        pos = [j for j in by_sentence[r.sentence_id]
               if j != i and sub.iloc[j].speaker_id != r.speaker_id]
        if not pos:
            continue
        # hard negatives: same speaker, different sentence
        same_spk = [j for j in by_speaker[r.speaker_id]
                    if j != i and sub.iloc[j].sentence_id != r.sentence_id]
        if len(same_spk) > n_negatives:
            same_spk = rng.choice(same_spk, size=n_negatives,
                                  replace=False).tolist()
        rows.append((i, pos, same_spk))
        candidates.update([i], pos, same_spk)

    if not rows:
        raise ValueError("no usable reranking queries")

    keep = sorted(candidates)
    pos_idx = {i: k for k, i in enumerate(keep)}
    kept = sub.iloc[keep].reset_index(drop=True)

    query_id, positive_ids = [], []
    qmap = {i: (sub.iloc[i].utt_id, [sub.iloc[j].utt_id for j in pos])
            for i, pos, _ in rows}
    for i in keep:
        if i in qmap:
            query_id.append(str(qmap[i][0]))
            positive_ids.append([str(x) for x in qmap[i][1]])
        else:
            query_id.append("")
            positive_ids.append([])

    n_pos = sum(len(p) for p in positive_ids)
    print(f"  {len(rows):,} queries, {len(keep):,} candidates, "
          f"{n_pos:,} positives "
          f"({n_pos/max(len(rows),1):.1f} per query)")

    features = Features({
        "audio": audio_feature(),
        "utt_id": Value("string"),
        "sentence_id": Value("string"),
        "speaker_id": Value("string"),
        "transcript": Value("string"),
        "query_id": Value("string"),
        "positive_ids": Sequence(Value("string")),
    })
    print(f"  decoding {len(kept):,} files...")
    ds = Dataset.from_dict(
        {
            "audio": [{"array": load_audio(p).tolist(), "sampling_rate": SR}
                      for p in kept.path],
            "utt_id": kept.utt_id.astype(str).tolist(),
            "sentence_id": kept.sentence_id.tolist(),
            "speaker_id": kept.speaker_id.astype(str).tolist(),
            "transcript": kept.transcript.astype(str).tolist(),
            "query_id": query_id,
            "positive_ids": positive_ids,
        },
        features=features,
    )
    return DatasetDict({"test": ds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/manifest.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "simteb_audio/data_run"))
    ap.add_argument("--groups", type=int, default=1200,
                    help="content groups for the retrieval corpus")
    ap.add_argument("--min-speakers", type=int, default=2)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--rerank-queries", type=int, default=500)
    ap.add_argument("--rerank-negatives", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["retrieval", "rerank"])
    a = ap.parse_args()

    manifest = pd.read_csv(a.manifest, sep="\t", quoting=3,
                           dtype={"speaker_id": str})

    print("=== selecting content groups")
    sub = pick_groups(manifest, a.groups, a.min_speakers,
                      a.min_dur, a.max_dur, a.seed)

    if "retrieval" not in a.skip:
        print("\n=== retrieval dataset (A2T / T2A / A2A)")
        dsd = build_retrieval(sub)
        ds = dsd["test"]
        assert len(set(ds["utt_id"])) == len(ds), "utt_id must be unique"
        assert ds[0]["audio"]["array"], "first example decoded empty"
        p = os.path.join(a.out, "openslr52_retrieval")
        dsd.save_to_disk(p)
        print(f"  saved {p}")

    if "rerank" not in a.skip:
        print("\n=== reranking dataset")
        dsd = build_rerank(sub, a.rerank_queries, a.rerank_negatives, a.seed)
        ds = dsd["test"]
        assert len(set(ds["utt_id"])) == len(ds), "utt_id must be unique"
        assert any(q for q in ds["query_id"]), "no queries marked"
        p = os.path.join(a.out, "openslr52_rerank")
        dsd.save_to_disk(p)
        print(f"  saved {p}")

    print(f"\nexport SIMTEB_AUDIO_LOCAL={a.out}")


if __name__ == "__main__":
    main()
