#!/usr/bin/env python3
"""Is the layer gain robust to seeds, and does it beat a no-selection baseline?

The layer-transfer result selected one layer on development data and compared it
against the final layer. Two objections follow, and both are answered here.

SEEDS. k-means is stochastic, so a single-seed gain of +0.572 could be partly
initialisation luck. Every score below is the mean over five seeds, and the
spread is reported.

A FAIRER BASELINE. "Final layer" is the library default, but it is a weak
comparator: it is one arbitrary layer. A practitioner who suspected the final
layer was wrong, without running a selection experiment, would more likely
average the layers. So the dev-selected layer is also compared against:

  final        the last hidden state (the default)
  uniform      embeddings averaged over ALL layers, then pooled
  best_test    the best layer measured on test data (an upper bound that no
               honest procedure can reach; it bounds what selection could win)

If the dev-selected layer beats the uniform average, layer selection is doing
real work rather than merely escaping a bad default.

Writes eval/analysis/Q_layer_seeds_uniform.tsv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SEEDS = (0, 1, 2, 3, 4)
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m",
         "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0",
         "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large",
         "hubert": "facebook/hubert-large-ls960-ft"}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def clean(a):
    return np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def ami_seeds(X, labels, seeds=SEEDS):
    """Mean and sd of AMI over seeds."""
    y = pd.factorize(labels)[0]
    k = len(np.unique(y))
    out = []
    for s in seeds:
        pred = MiniBatchKMeans(n_clusters=k, random_state=s, n_init=5,
                               batch_size=1024).fit_predict(l2(X))
        out.append(adjusted_mutual_info_score(y, pred))
    return float(np.mean(out)), float(np.std(out))


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str}).set_index("utt_id")
    J = pd.read_csv(os.path.join(ROOT, "eval/analysis/J_layer_transfer.tsv"),
                    sep="\t")
    picked = {r.tag: int(r.picked_layer)
              for _, r in J[J.capability == "speaker"].iterrows()}

    rows = []
    for tag in sorted(picked):
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]
        test = np.flatnonzero(meta.dataset.values == "openslr30")
        y = meta.speaker_id.values[test]
        n_layers = emb.shape[0]
        final = n_layers - 1
        pick = picked[tag]

        sel_m, sel_s = ami_seeds(clean(emb[pick])[test], y)
        fin_m, fin_s = ami_seeds(clean(emb[final])[test], y)
        # uniform: mean of the per-layer pooled embeddings
        uni = clean(emb.mean(axis=0))[test]
        uni_m, uni_s = ami_seeds(uni, y)

        # upper bound: best layer judged on test (not an honest procedure)
        per_layer = [ami_seeds(clean(emb[li])[test], y, seeds=(0,))[0]
                     for li in range(n_layers)]
        best_li = int(np.argmax(per_layer))

        rows.append(dict(
            model=NAMES.get(tag, tag), tag=tag, n_layers=n_layers,
            n_test=int(test.size), n_speakers=int(len(set(y.tolist()))),
            picked_layer=pick, final_layer=final, best_test_layer=best_li,
            sel_mean=sel_m, sel_sd=sel_s,
            final_mean=fin_m, final_sd=fin_s,
            uniform_mean=uni_m, uniform_sd=uni_s,
            best_test_mean=per_layer[best_li],
            gain_vs_final=sel_m - fin_m,
            gain_vs_uniform=sel_m - uni_m,
            shortfall_vs_oracle=per_layer[best_li] - sel_m,
        ))
        print(f"  {NAMES.get(tag, tag):34s} sel {sel_m:.3f}±{sel_s:.3f}  "
              f"final {fin_m:.3f}±{fin_s:.3f}  uniform {uni_m:.3f}±{uni_s:.3f}  "
              f"oracle {per_layer[best_li]:.3f} (L{best_li})")
        del emb, z

    df = pd.DataFrame(rows).sort_values("gain_vs_uniform", ascending=False)
    out = os.path.join(ROOT, "eval/analysis/Q_layer_seeds_uniform.tsv")
    df.to_csv(out, sep="\t", index=False)

    print("\n## Q. Layer gains over 5 seeds, against two baselines")
    print(f"   test = OpenSLR-30, {df.n_test.iloc[0]} utts, "
          f"{df.n_speakers.iloc[0]} speakers; layer chosen on OpenSLR-52\n")
    print(df[["model", "picked_layer", "sel_mean", "sel_sd", "final_mean",
              "uniform_mean", "best_test_mean", "gain_vs_final",
              "gain_vs_uniform", "shortfall_vs_oracle"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  seed sd of the selected layer: max "
          f"{df.sel_sd.max():.3f}, median {df.sel_sd.median():.3f}")
    print(f"  beats the final layer:   {int((df.gain_vs_final > 0).sum())}/{len(df)}"
          f"  median {df.gain_vs_final.median():+.3f}")
    print(f"  beats the uniform mean:  {int((df.gain_vs_uniform > 0).sum())}/{len(df)}"
          f"  median {df.gain_vs_uniform.median():+.3f}")
    print(f"  shortfall against a test-chosen layer: median "
          f"{df.shortfall_vs_oracle.median():+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
