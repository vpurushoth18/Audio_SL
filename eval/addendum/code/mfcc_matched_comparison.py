#!/usr/bin/env python3
"""MFCC against encoders under a strictly identical clustering protocol.

The earlier comparison (P_mfcc_vs_layer) is not fair: the MFCC rows were
clustered with n_init=5, batch_size=512 while the encoder rows were carried
over from layer_transfer.py at n_init=3, batch_size=1024, and the two used
different seed counts. The 0.55x / 2.00x ratios it reports therefore mix a
genuine effect with a protocol difference.

This recomputes everything in one pass with no shared state between arms
except the settings:

  same utterances, same speaker labels
  same seeds (0,1,2,3,4)
  same MiniBatchKMeans configuration for every arm
  layer selection still made on DEVELOPMENT data only

Arms: MFCC-40, final layer, development-selected layer.

Writes eval/analysis/mfcc_matched_comparison.tsv (per-seed) and a summary.
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
sys.path.insert(0, os.path.join(ROOT, "simteb_audio/src"))
from simteb_audio.models.mfcc_baseline import MFCCBaselineWrapper  # noqa: E402

# one configuration, used by every arm
SEEDS = (0, 1, 2, 3, 4)
KM = dict(n_init=5, batch_size=1024, max_iter=100, tol=0.0, max_no_improvement=10)

NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def ami_per_seed(X, labels):
    y = pd.factorize(labels)[0]
    k = len(np.unique(y))
    out = []
    for s in SEEDS:
        pred = MiniBatchKMeans(n_clusters=k, random_state=s, **KM).fit_predict(l2(X))
        out.append(float(adjusted_mutual_info_score(y, pred)))
    return out


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str})
    J = pd.read_csv(os.path.join(ROOT, "eval/analysis/J_layer_transfer.tsv"), sep="\t")
    picked = {r.tag: int(r.picked_layer) for _, r in J[J.capability == "speaker"].iterrows()}

    # the evaluation set: OpenSLR-30, exactly as the layer tables use it
    s30 = sub[sub.dataset == "openslr30"].copy()
    print(f"  evaluation set: {len(s30)} utterances, {s30.speaker_id.nunique()} speakers")
    print(f"  configuration : seeds {SEEDS}, {KM}\n")

    rows = []

    # ---- MFCC arm, computed from audio with the same feature code as the benchmark
    w = MFCCBaselineWrapper()
    X = []
    for p in s30.path:
        x, sr = sf.read(os.path.join(ROOT, p), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            x = soxr.resample(x, sr, 16000)
        X.append(w._features(x))
    mf = ami_per_seed(np.stack(X), s30.speaker_id.values)
    for s, v in zip(SEEDS, mf):
        rows.append(dict(arm="mfcc40", model="baseline/mfcc-40", layer=np.nan,
                         seed=s, ami=v))
    print(f"  {'baseline/mfcc-40':34s} MFCC      {np.mean(mf):.4f} +/- {np.std(mf):.4f}",
          flush=True)

    # ---- encoder arms
    for tag in sorted(NAMES):
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.set_index("utt_id").loc[utt]
        idx = np.flatnonzero(meta.dataset.values == "openslr30")
        y = meta.speaker_id.values[idx]
        assert len(idx) == len(s30), "evaluation set must match the MFCC arm"
        for arm, li in (("final", emb.shape[0] - 1), ("selected", picked[tag])):
            E = np.nan_to_num(emb[li].astype(np.float32))[idx]
            vals = ami_per_seed(E, y)
            for s, v in zip(SEEDS, vals):
                rows.append(dict(arm=arm, model=NAMES[tag], layer=li, seed=s, ami=v))
            print(f"  {NAMES[tag]:34s} {arm:9s} {np.mean(vals):.4f} +/- "
                  f"{np.std(vals):.4f}", flush=True)
        del emb, z

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "eval/analysis/mfcc_matched_per_seed.tsv"),
             sep="\t", index=False)

    # ---- summary with recomputed ratios
    g = d.groupby(["arm", "model", "layer"], dropna=False).ami.agg(["mean", "std"]).reset_index()
    mfcc_mean = float(g[g.arm == "mfcc40"]["mean"].iloc[0])
    enc = g[g.arm != "mfcc40"].copy()
    enc["mfcc_mean"] = mfcc_mean
    enc["ratio_encoder_over_mfcc"] = enc["mean"] / mfcc_mean
    enc = enc.sort_values(["arm", "ratio_encoder_over_mfcc"], ascending=[True, False])
    out = os.path.join(ROOT, "eval/analysis/mfcc_matched_comparison.tsv")
    enc.to_csv(out, sep="\t", index=False)

    print(f"\n  MFCC mean AMI = {mfcc_mean:.4f}\n")
    for arm in ("final", "selected"):
        a = enc[enc.arm == arm]
        print(f"  {arm:9s}: encoders above MFCC {int((a['mean'] > mfcc_mean).sum())}/{len(a)}  "
              f"median ratio {a.ratio_encoder_over_mfcc.median():.2f}x  "
              f"range {a.ratio_encoder_over_mfcc.min():.2f}-{a.ratio_encoder_over_mfcc.max():.2f}x")
    print(f"\nwrote {out}")
    print(f"wrote {ROOT}/eval/analysis/mfcc_matched_per_seed.tsv")


if __name__ == "__main__":
    main()
