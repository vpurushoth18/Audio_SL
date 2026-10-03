#!/usr/bin/env python3
"""MFCC against encoders under identical conditions, at two layer choices.

The first version of this comparison was invalid: it set a V-measure MFCC score
against an AMI score taken from a development-selected layer, so the metric and
the layer condition both differed. The conclusion drawn from it — that speaker
tasks do not need a learned representation — did not follow.

This recomputes both sides on the same task, with the same metric, the same
k-means implementation and the same seeds, and separates the two layer
conditions that were previously conflated:

  final layer      what the benchmark actually used
  selected layer   chosen on OpenSLR-52 development data, frozen

The answer changes with the layer condition, which is the finding.

Writes eval/analysis/P_mfcc_vs_layer.tsv.
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
sys.path.insert(0, os.path.join(ROOT, "simteb_audio/src"))
from simteb_audio.models.mfcc_baseline import MFCCBaselineWrapper  # noqa: E402

SEEDS = (0, 1, 2)


def cluster_scores(X, labels, seeds=SEEDS):
    """Mean AMI and V-measure over seeds, L2-normalised features."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    y = pd.factorize(labels)[0]
    k = len(np.unique(y))
    a, v = [], []
    for s in seeds:
        pred = MiniBatchKMeans(n_clusters=k, random_state=s, n_init=5,
                               batch_size=512).fit_predict(Xn)
        a.append(adjusted_mutual_info_score(y, pred))
        v.append(v_measure_score(y, pred))
    return float(np.mean(a)), float(np.std(a)), float(np.mean(v)), k


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t",
                      quoting=3, dtype={"speaker_id": str})
    w = MFCCBaselineWrapper()

    rows = []
    for ds in ("openslr30", "openslr52"):
        s = sub[sub.dataset == ds]
        X = []
        for p in s.path:
            x, sr = sf.read(os.path.join(ROOT, p), dtype="float32",
                            always_2d=False)
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                x = soxr.resample(x, sr, 16000)
            X.append(w._features(x))
        ami, sd, vm, k = cluster_scores(np.stack(X), s.speaker_id.values)
        rows.append(dict(corpus=ds, n=len(s), k=k, model="baseline/mfcc-40",
                         layer_condition="n/a (no layers)", ami=ami, ami_sd=sd,
                         v_measure=vm))
        print(f"{ds}: n={len(s)} k={k}  MFCC AMI={ami:.3f} (sd {sd:.3f})  "
              f"V={vm:.3f}  inflation={vm - ami:+.3f}")

    # encoder side comes from the frozen layer-transfer run on the same task
    J = pd.read_csv(os.path.join(ROOT, "eval/analysis/J_layer_transfer.tsv"),
                    sep="\t")
    spk = J[J.capability == "speaker"]
    mfcc30 = [r for r in rows if r["corpus"] == "openslr30"][0]["ami"]
    for _, r in spk.iterrows():
        for cond, col, lay in (("final layer", "test_score_final",
                                int(r.n_layers) - 1),
                               ("selected layer", "test_score_picked",
                                int(r.picked_layer))):
            rows.append(dict(corpus="openslr30", n=int(r.n_test), k=12,
                             model=r.model, layer_condition=f"{cond} ({lay})",
                             ami=float(r[col]), ami_sd=np.nan,
                             v_measure=np.nan,
                             ratio_to_mfcc=float(r[col]) / mfcc30))

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/P_mfcc_vs_layer.tsv")
    df.to_csv(out, sep="\t", index=False)

    enc = df[(df.model != "baseline/mfcc-40") & (df.corpus == "openslr30")]
    fin = enc[enc.layer_condition.str.startswith("final")]
    sel = enc[enc.layer_condition.str.startswith("selected")]
    print(f"\nOpenSLR-30 speaker AMI, MFCC = {mfcc30:.3f}")
    print(f"  encoders above MFCC at the final layer:    "
          f"{int((fin.ami > mfcc30).sum())}/{len(fin)}  "
          f"(median ratio {fin.ratio_to_mfcc.median():.2f}x)")
    print(f"  encoders above MFCC at the selected layer: "
          f"{int((sel.ami > mfcc30).sum())}/{len(sel)}  "
          f"(median ratio {sel.ratio_to_mfcc.median():.2f}x)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
