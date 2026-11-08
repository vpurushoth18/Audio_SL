#!/usr/bin/env python3
"""Benchmark audit: does each task measure what the paper will claim?

Five checks, each able to demote or remove a task. Run before interpreting any
ranking.

  1. Prompt-speaker confound   can a content task be solved by voice?
  2. Duplicate audio           within and across corpora
  3. Shortcut baseline         can pooled MFCCs solve the task?
  4. Chance baselines          majority / positive-rate / label-permutation
  5. Label provenance          human, derived, or automatic

The MFCC baseline is the important one. It has no learned component, so if it
matches the encoders on a task, that task measures recording characteristics
rather than representation quality.

Writes eval/audit/*.tsv and a verdict table.
"""

import argparse
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             average_precision_score, v_measure_score)

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DATA = os.path.join(ROOT, "simteb_audio/data_run")
RNG = np.random.default_rng(0)
SR = 16000

# Label provenance, read off each corpus's documentation.
PROVENANCE = {
    "OpenSLR52SpeakerClustering": ("speaker_id", "human", "OpenSLR-52 speaker ids"),
    "OpenSLR30SpeakerClustering": ("speaker_id", "human", "OpenSLR-30 speaker ids"),
    "OpenSLR52SpeakerClassification": ("speaker_id", "human", "OpenSLR-52"),
    "OpenSLR30SpeakerClassification": ("speaker_id", "human", "OpenSLR-30"),
    "OpenSLR52SpeakerPairClassification": (
        "speaker_id", "derived", "pairs built from human speaker ids"),
    "OpenSLR30SpeakerPairClassification": (
        "speaker_id", "derived", "pairs built from human speaker ids"),
    "OpenSLR52AudioRetrieval": (
        "transcript", "derived", "content groups from human transcripts"),
    "OpenSLR52AudioReranking": (
        "transcript", "derived", "content groups + same-speaker hard negatives"),
    "OmnilingualSinhalaSpeakerClustering": ("speaker_id", "human", "Omnilingual"),
    "OmnilingualSinhalaSpeakerClassification": ("speaker_id", "human", "Omnilingual"),
    "OmnilingualSinhalaContentRetrieval": (
        "prompt_id", "human", "Omnilingual prompt ids"),
    "WorldSpeechSinhalaSessionClustering": (
        "session_date", "metadata", "sitting day, NOT a speaker label"),
    "WorldSpeechSinhalaQualityClassification": (
        "dnsmos_ovr", "derived-by-us", "DNSMOS terciles chosen here"),
    "VoxLingua107SinhalaVideoClustering": (
        "video_id", "automatic", "YouTube id; a video may hold several speakers"),
}

# Which prepared dataset backs each scored task, and its label column.
BACKING = {
    "OpenSLR52SpeakerClustering": ("openslr52_speaker", "speaker_id", "clustering"),
    "OpenSLR30SpeakerClustering": ("openslr30_speaker", "speaker_id", "clustering"),
    "OmnilingualSinhalaSpeakerClustering": (
        "omnilingual_sin_speaker", "speaker_id", "clustering"),
    "WorldSpeechSinhalaSessionClustering": (
        "worldspeech_sin_session", "session_id", "clustering"),
    "VoxLingua107SinhalaVideoClustering": (
        "voxlingua_sin_video", "video_id", "clustering"),
    "OpenSLR52SpeakerClassification": (
        "openslr52_speaker_clf", "speaker_id", "classification"),
    "OpenSLR30SpeakerClassification": (
        "openslr30_speaker_clf", "speaker_id", "classification"),
    "OmnilingualSinhalaSpeakerClassification": (
        "omnilingual_sin_speaker_clf", "speaker_id", "classification"),
    "WorldSpeechSinhalaQualityClassification": (
        "worldspeech_sin_quality", "quality_band", "classification"),
    "OpenSLR52SpeakerPairClassification": (
        "openslr52_speaker_pairs", "labels", "pair"),
    "OpenSLR30SpeakerPairClassification": (
        "openslr30_speaker_pairs", "labels", "pair"),
}


# ------------------------------------------------------------------ features
def mfcc_stats(array, sr=SR, n_mfcc=20, n_mels=40):
    """Pooled MFCC mean+std: 40 dims, no learned component.

    librosa is absent, so this is self-contained: frame -> power spectrum ->
    mel filterbank -> log -> DCT-II, then mean and std over frames.
    """
    x = np.asarray(array, dtype=np.float32)
    n_fft, hop = 512, 160
    if x.size < n_fft * 2:
        x = np.pad(x, (0, n_fft * 2 - x.size))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2

    f_hz = np.linspace(0, sr / 2, spec.shape[1])
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)       # noqa: E731
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)     # noqa: E731
    edges = mel2hz(np.linspace(hz2mel(20), hz2mel(sr / 2), n_mels + 2))
    fb = np.zeros((n_mels, spec.shape[1]), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (f_hz >= lo) & (f_hz <= mid)
        right = (f_hz > mid) & (f_hz <= hi)
        fb[i, left] = (f_hz[left] - lo) / max(mid - lo, 1e-9)
        fb[i, right] = (hi - f_hz[right]) / max(hi - mid, 1e-9)
    mel = np.log(spec @ fb.T + 1e-10)

    k = np.arange(n_mels)
    dct = np.cos(np.pi / n_mels * (k[None, :] + 0.5) * np.arange(n_mfcc)[:, None])
    c = mel @ dct.T
    return np.concatenate([c.mean(0), c.std(0)]).astype(np.float32)


def task_features(dirname, label_col, kind, limit=None):
    path = os.path.join(DATA, dirname)
    if not os.path.isdir(path):
        return None
    dsd = load_from_disk(path)
    out = {}
    for split, ds in dsd.items():
        n = len(ds) if limit is None else min(limit, len(ds))
        if kind == "pair":
            X1 = np.stack([mfcc_stats(ds[i]["audio1"]["array"]) for i in range(n)])
            X2 = np.stack([mfcc_stats(ds[i]["audio2"]["array"]) for i in range(n)])
            out[split] = (X1, X2, np.asarray(list(ds[label_col])[:n]))
        else:
            X = np.stack([mfcc_stats(ds[i]["audio"]["array"]) for i in range(n)])
            out[split] = (X, np.asarray(list(ds[label_col])[:n]))
    return out


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


# ------------------------------------------- 1. prompt-speaker confound
def check_confound(rep, out):
    rows = []
    for task, dirname in [
        ("OpenSLR52AudioRetrieval", "openslr52_retrieval"),
        ("OpenSLR52AudioReranking", "openslr52_rerank"),
        ("OmnilingualSinhalaContentRetrieval", "omnilingual_sin_retrieval"),
    ]:
        p = os.path.join(DATA, dirname)
        if not os.path.isdir(p):
            continue
        ds = load_from_disk(p)["test"]
        # list(...) because datasets 4.x returns a Column object
        sids = list(ds["sentence_id"])
        spks = list(ds["speaker_id"])
        by_s = defaultdict(list)
        for s, k in zip(sids, spks):
            by_s[s].append(k)
        nspk = [len(set(v)) for v in by_s.values()]
        # Does prompt identity carry information about speaker identity?
        ami = adjusted_mutual_info_score(
            pd.factorize(pd.Series(sids))[0], pd.factorize(pd.Series(spks))[0])
        rows.append(dict(
            task=task, n_utts=len(ds), n_prompts=len(by_s),
            n_speakers=len(set(spks)),
            spk_per_prompt_median=float(np.median(nspk)),
            spk_per_prompt_max=int(max(nspk)),
            prompts_single_speaker=int(sum(1 for x in nspk if x == 1)),
            ami_prompt_vs_speaker=ami))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "1_prompt_speaker_confound.tsv"), sep="\t",
              index=False)
    rep.append("\n## 1. Prompt-speaker confound\n")
    rep.append(df.to_string(index=False))
    rep.append(
        "\n  ami_prompt_vs_speaker is adjusted mutual information between prompt\n"
        "  identity and speaker identity. Near 0 means the prompt tells you\n"
        "  nothing about who spoke it, so a cross-speaker content task cannot be\n"
        "  solved by voice. Above ~0.3 the content tasks are confounded.")
    return df


# ------------------------------------------------------- 2. duplicate audio
def check_duplicates(rep, out, limit):
    rows, where = [], defaultdict(set)
    for dirname in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, dirname)
        if not os.path.isdir(p):
            continue
        try:
            dsd = load_from_disk(p)
        except Exception:
            continue
        seen, n = Counter(), 0
        for split, ds in dsd.items():
            keys = [c for c in ("audio", "audio1", "audio2")
                    if c in ds.column_names]
            m = len(ds) if limit is None else min(limit, len(ds))
            for i in range(m):
                row = ds[i]
                for k in keys:
                    a = np.asarray(row[k]["array"], dtype=np.float32)
                    h = hashlib.md5(a.tobytes()).hexdigest()
                    seen[h] += 1
                    where[h].add(dirname)
                    n += 1
        dup = sum(v - 1 for v in seen.values() if v > 1)
        rows.append(dict(dataset=dirname, n_clips=n, n_unique=len(seen),
                         exact_duplicates=dup,
                         pct_duplicate=100.0 * dup / max(n, 1)))
    cross = Counter(tuple(sorted(d)) for d in where.values() if len(d) > 1)
    df = pd.DataFrame(rows).sort_values("pct_duplicate", ascending=False)
    df.to_csv(os.path.join(out, "2_duplicates.tsv"), sep="\t", index=False)
    rep.append("\n\n## 2. Duplicate audio\n")
    rep.append(df.to_string(index=False))
    rep.append(f"\n  clips present in more than one prepared dataset: "
               f"{sum(cross.values())}")
    for k, v in cross.most_common(10):
        rep.append(f"    {v:6d}  {' + '.join(k)}")
    rep.append("  Overlap BETWEEN datasets treated as separate corpora would\n"
               "  undermine cross-corpus claims. Overlap WITHIN one dataset is\n"
               "  expected where a pair task reuses clips by design.")
    return df


# ----------------------------------- 3 + 4. MFCC shortcut and chance baselines
def check_shortcuts(rep, out, limit):
    rows = []
    for task, (dirname, label_col, kind) in BACKING.items():
        feats = task_features(dirname, label_col, kind, limit=limit)
        if feats is None:
            rows.append(dict(task=task, note="prepared dataset missing"))
            continue
        if kind == "clustering":
            X, y = feats["test"]
            k = len(set(y.tolist()))
            pred = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                                   batch_size=1024).fit_predict(l2(X))
            perm = [v_measure_score(RNG.permutation(y), pred) for _ in range(20)]
            rows.append(dict(task=task, kind=kind, metric="v_measure",
                             n=len(y), k=k,
                             mfcc_score=v_measure_score(y, pred),
                             mfcc_ami=adjusted_mutual_info_score(y, pred),
                             chance=float(np.mean(perm)),
                             chance_sd=float(np.std(perm))))
        elif kind == "classification":
            if not {"train", "test"} <= set(feats):
                rows.append(dict(task=task, note="no train/test split"))
                continue
            Xtr, ytr = feats["train"]
            Xte, yte = feats["test"]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, ytr)
            rows.append(dict(task=task, kind=kind, metric="accuracy",
                             n=len(yte), k=len(set(ytr.tolist())),
                             mfcc_score=accuracy_score(
                                 yte, clf.predict((Xte - mu) / sd)),
                             chance=Counter(ytr.tolist()).most_common(1)[0][1]
                             / len(ytr)))
        else:  # pair
            X1, X2, y = feats["test"]
            sim = (l2(X1) * l2(X2)).sum(1)
            rows.append(dict(task=task, kind=kind, metric="max_ap", n=len(y), k=2,
                             mfcc_score=average_precision_score(y, sim),
                             chance=float(np.mean(y))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "3_mfcc_and_chance.tsv"), sep="\t", index=False)
    rep.append("\n\n## 3+4. MFCC shortcut and chance baselines\n")
    rep.append(df.to_string(index=False))
    rep.append(
        "\n  mfcc_score uses a 40-dim pooled-MFCC feature with no learned\n"
        "  component. Close to the encoders means the task is solvable from\n"
        "  recording characteristics. `chance` is label-permuted V-measure\n"
        "  (clustering), the majority class (classification), or the positive\n"
        "  rate (pairs).")
    return df


# ---------------------------------------------------------- 5. provenance
def check_provenance(rep, out):
    df = pd.DataFrame([dict(task=t, label=l, provenance=p, note=n)
                       for t, (l, p, n) in PROVENANCE.items()])
    df.to_csv(os.path.join(out, "5_label_provenance.tsv"), sep="\t", index=False)
    rep.append("\n\n## 5. Label provenance\n")
    rep.append(df.to_string(index=False))
    rep.append("\n  'automatic' and 'metadata' labels must not feed a SPEAKER\n"
               "  capability score: a VoxLingua video or a WorldSpeech sitting\n"
               "  day can contain several speakers.")
    return df


# ------------------------------------------------------------ 6. verdicts
def verdicts(short, results_prefix, rep, out):
    per_task = pd.read_csv(f"{results_prefix}_per_task.tsv", sep="\t")
    # the per-task table is x100; bring it back to the 0-1 scale of the audit
    best = {c: per_task[c].max() / 100.0
            for c in per_task.columns if c != "model"}

    rows = []
    for task, (label, prov, note) in PROVENANCE.items():
        s = short[short.task == task]
        g = lambda col: (float(s[col].iloc[0])                     # noqa: E731
                         if len(s) and col in s and pd.notna(s[col].iloc[0])
                         else np.nan)
        mfcc, chance = g("mfcc_score"), g("chance")
        top = best.get(task, np.nan)

        verdict, reason = "core", []
        if np.isfinite(mfcc) and np.isfinite(top) and top > 0:
            frac = mfcc / top
            if frac > 0.90:
                verdict = "diagnostic"
                reason.append(f"MFCC reaches {100 * frac:.0f}% of best encoder")
        if np.isfinite(chance) and np.isfinite(top) and (top - chance) < 0.05:
            verdict = "exclude"
            reason.append("best encoder within 0.05 of chance")
        if prov in ("automatic", "metadata"):
            reason.append(f"{prov} label: not a speaker capability")
            verdict = "diagnostic" if verdict == "core" else verdict
        if prov == "derived-by-us":
            reason.append("label thresholds chosen by us")
            verdict = "diagnostic" if verdict == "core" else verdict
        rows.append(dict(task=task, provenance=prov, best_encoder=top,
                         mfcc=mfcc, chance=chance, verdict=verdict,
                         reason="; ".join(reason) or "passes"))
    df = pd.DataFrame(rows).sort_values(["verdict", "task"])
    df.to_csv(os.path.join(out, "6_verdicts.tsv"), sep="\t", index=False)
    rep.append("\n\n## 6. Verdict per task\n")
    rep.append(df.to_string(index=False))
    rep.append(f"\n  core {sum(df.verdict == 'core')} | "
               f"diagnostic {sum(df.verdict == 'diagnostic')} | "
               f"exclude {sum(df.verdict == 'exclude')}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/audit"))
    ap.add_argument("--results",
                    default=os.path.join(ROOT, "eval/simteb_audio_results"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap clips per split for a fast pass")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["confound", "dups", "shortcuts", "prov"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rep = ["# Benchmark audit\n"]
    short = None
    if "confound" not in a.skip:
        check_confound(rep, a.out)
    if "dups" not in a.skip:
        check_duplicates(rep, a.out, a.limit)
    if "shortcuts" not in a.skip:
        short = check_shortcuts(rep, a.out, a.limit)
    if "prov" not in a.skip:
        check_provenance(rep, a.out)
    if short is not None:
        verdicts(short, a.results, rep, a.out)

    text = "\n".join(str(x) for x in rep)
    Path(os.path.join(a.out, "audit.md")).write_text(text)
    print(text)
    print(f"\nwrote {a.out}/audit.md and the per-check TSVs")


if __name__ == "__main__":
    main()
