#!/usr/bin/env python3
"""Second corpus for the layer recommendation: does it also hold on Omnilingual?

The main transfer test chose a layer on OpenSLR-52 and evaluated on OpenSLR-30.
Both are read-speech corpora collected the same way, so agreement between them
is a weak test of generality. Omnilingual Sinhala is a different collection with
different speakers and recording conditions, which makes it a harder test of the
same recommendation.

Procedure, identical to the main test apart from the test corpus:

  1. choose the layer on OpenSLR-52 development data (speaker AMI),
  2. freeze it,
  3. evaluate on Omnilingual,
  4. compare against the default final layer.

Omnilingual has only 7 speakers, so AMI at k=7 is a coarser measurement than the
12-speaker OpenSLR-30 test. A positive gain here is therefore evidence that the
early-layer recommendation survives a corpus change, not a precise effect size.

Writes eval/analysis/O_layer_transfer_omnilingual.tsv.
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m",
         "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0",
         "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large",
         "hubert": "facebook/hubert-large-ls960-ft"}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def ami(E, labels, seeds=(0, 1, 2)):
    """Mean AMI over a few k-means seeds; 7 clusters is sensitive to init."""
    y = pd.factorize(labels)[0]
    k = len(np.unique(y))
    if k < 2:
        return np.nan
    out = []
    for s in seeds:
        pred = MiniBatchKMeans(n_clusters=k, random_state=s, n_init=5,
                               batch_size=512).fit_predict(l2(E))
        out.append(adjusted_mutual_info_score(y, pred))
    return float(np.mean(out))


def clean(a):
    return np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    dev_sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t",
                          quoting=3, dtype={"speaker_id": str}).set_index("utt_id")
    te_sub = pd.read_csv(os.path.join(ROOT, "eval/subset_omni.tsv"), sep="\t",
                         quoting=3, dtype={"speaker_id": str}).set_index("utt_id")

    rows = []
    for tag in sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "eval/emb_omni"))
                      if f.endswith(".npz")):
        zd = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        zt = np.load(os.path.join(ROOT, f"eval/emb_omni/{tag}.npz"), allow_pickle=True)
        dev_emb, dev_utt = zd["emb"], zd["utt"].astype(str)
        te_emb, te_utt = zt["emb"], zt["utt"].astype(str)

        dmeta = dev_sub.loc[dev_utt]
        tmeta = te_sub.loc[te_utt]
        dev_idx = np.flatnonzero(dmeta.dataset.values == "openslr52")
        n_layers = dev_emb.shape[0]
        final = n_layers - 1

        # sweep layers on OpenSLR-52 only
        dev_scores = [ami(clean(dev_emb[li])[dev_idx],
                          dmeta.speaker_id.values[dev_idx], seeds=(0,))
                      for li in range(n_layers)]
        pick = int(np.nanargmax(dev_scores))

        rows.append(dict(
            model=NAMES.get(tag, tag), tag=tag, n_layers=n_layers,
            dev="OpenSLR-52", test="Omnilingual Sinhala",
            n_dev=int(dev_idx.size), n_test=len(te_utt),
            n_test_speakers=int(tmeta.speaker_id.nunique()),
            picked_layer=pick, picked_depth=pick / max(final, 1),
            dev_score=dev_scores[pick],
            test_score_picked=ami(clean(te_emb[pick]), tmeta.speaker_id.values),
            test_score_final=ami(clean(te_emb[final]), tmeta.speaker_id.values),
        ))
        del dev_emb, te_emb, zd, zt

    df = pd.DataFrame(rows)
    df["gain_over_final"] = df.test_score_picked - df.test_score_final
    df = df.sort_values("gain_over_final", ascending=False)
    out = os.path.join(ROOT, "eval/analysis/O_layer_transfer_omnilingual.tsv")
    df.to_csv(out, sep="\t", index=False)

    print("\n## O. Layer transfer, second test corpus")
    print(f"   dev = OpenSLR-52 ({df.n_dev.iloc[0]:,} utts, read speech)")
    print(f"   test = Omnilingual Sinhala ({df.n_test.iloc[0]} utts, "
          f"{df.n_test_speakers.iloc[0]} speakers)")
    print("   metric = speaker AMI, mean over 3 k-means seeds\n")
    print(df[["model", "picked_layer", "picked_depth", "dev_score",
              "test_score_picked", "test_score_final", "gain_over_final"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    w = int((df.gain_over_final > 0).sum())
    print(f"\n  the dev-chosen layer beats the final layer on {w}/{len(df)} models; "
          f"median gain {df.gain_over_final.median():+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
