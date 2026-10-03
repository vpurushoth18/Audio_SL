#!/usr/bin/env python3
"""Does the early-layer recommendation hold for CLASSIFICATION, not just clustering?

The layer study selected layers by speaker-clustering AMI and the recommendation
table then carried that depth over to classification, untested. That is the
softest step in the argument, so it is tested here directly.

Same protocol as the clustering version, with a probe in place of k-means:

  1. sweep layers on OpenSLR-52 development data, scoring by probe accuracy,
  2. freeze the chosen layer,
  3. evaluate on two held-out corpora: OpenSLR-30 and Omnilingual Sinhala,
  4. compare against the final layer, the all-layer average, and the layer that
     clustering AMI would have chosen.

The last column is the one that matters. If the clustering-selected layer scores
close to the classification-selected layer, carrying the depth across task types
is justified. If not, the recommendation needs splitting by task type.

Speakers are disjoint between the probe's train and test splits only for the
cross-corpus evaluations; within OpenSLR-52 the probe is trained and scored on
the same speaker inventory, which is what speaker classification means.

Writes eval/analysis/T_layer_transfer_classification.tsv.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SEEDS = (0, 1, 2)
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def clean(a):
    return np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fit_probe_gpu(Xtr, ytr, Xte, yte, n_classes, l2=1.0, steps=200):
    """Multinomial logistic regression on the GPU, full batch, LBFGS.

    sklearn's solver is CPU-only, and a 476-class fit over 11k samples takes
    minutes, which makes a per-layer sweep impractical: the first attempt at
    this experiment saturated 58 cores while both GPUs sat idle. This is the
    same model - linear map, softmax, cross-entropy, L2 penalty - solved with
    LBFGS on the GPU instead.

    These accuracies are used only to rank layers against each other within
    this experiment, where every layer is fitted identically. They are not
    interchangeable with the benchmark's own sklearn-based classification
    scores, which use MTEB's protocol.
    """
    Xtr = torch.as_tensor(Xtr, device=DEVICE, dtype=torch.float32)
    Xte = torch.as_tensor(Xte, device=DEVICE, dtype=torch.float32)
    ytr = torch.as_tensor(ytr, device=DEVICE, dtype=torch.long)
    yte = torch.as_tensor(yte, device=DEVICE, dtype=torch.long)

    W = torch.zeros(Xtr.shape[1], n_classes, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_classes, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, history_size=10,
                            line_search_fn="strong_wolfe")
    lossf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(Xtr @ W + b, ytr) + l2 / (2 * len(Xtr)) * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = ((Xte @ W + b).argmax(1) == yte).float().mean().item()
    del Xtr, Xte, ytr, yte, W, b
    return acc


def probe(X, y, seeds=SEEDS, test_size=0.3):
    """Mean accuracy of the GPU logistic probe over stratified splits.

    Classes with a single example cannot be stratified, so they are dropped,
    and the kept count is returned so the denominator stays explicit.
    """
    keep = pd.Series(y).groupby(y).transform("size").values >= 2
    X, y = X[keep], np.asarray(y)[keep]
    codes, uniq = pd.factorize(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = ((X - mu) / sd).astype(np.float32)
    out = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(Xs, codes, test_size=test_size,
                                              random_state=s, stratify=codes)
        out.append(_fit_probe_gpu(Xtr, ytr, Xte, yte, len(uniq)))
    return float(np.mean(out)), float(np.std(out)), int(len(y)), int(len(uniq))


def main():
    dev_sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t",
                          quoting=3, dtype={"speaker_id": str}).set_index("utt_id")
    omni_sub = pd.read_csv(os.path.join(ROOT, "eval/subset_omni.tsv"), sep="\t",
                           quoting=3, dtype={"speaker_id": str}).set_index("utt_id")
    J = pd.read_csv(os.path.join(ROOT, "eval/analysis/J_layer_transfer.tsv"),
                    sep="\t")
    clust_pick = {r.tag: int(r.picked_layer)
                  for _, r in J[J.capability == "speaker"].iterrows()}

    rows = []
    for tag in sorted(NAMES):
        f = os.path.join(ROOT, f"eval/emb/{tag}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = dev_sub.loc[utt]
        dev = np.flatnonzero(meta.dataset.values == "openslr52")
        t30 = np.flatnonzero(meta.dataset.values == "openslr30")
        n_layers, final = emb.shape[0], emb.shape[0] - 1

        # --- 1. sweep on dev, scoring by probe accuracy (one seed, for speed)
        dev_y = meta.speaker_id.values[dev]
        dev_scores = [probe(clean(emb[li])[dev], dev_y, seeds=(0,))[0]
                      for li in range(n_layers)]
        pick = int(np.argmax(dev_scores))
        print(f"  {NAMES[tag]:34s} clf-picked L{pick} (dev acc "
              f"{dev_scores[pick]:.3f}) | clustering picked L{clust_pick[tag]}",
              flush=True)

        # --- 2-4. evaluate frozen choices on the two held-out corpora
        zo = np.load(os.path.join(ROOT, f"eval/emb_omni/{tag}.npz"),
                     allow_pickle=True)
        oemb, outt = zo["emb"], zo["utt"].astype(str)
        om = omni_sub.loc[outt]

        for corpus, E, idx, y in (
                ("OpenSLR-30", emb, t30, meta.speaker_id.values[t30]),
                ("Omnilingual", oemb, np.arange(len(outt)), om.speaker_id.values)):
            def at(li, uniform=False):
                X = (clean(E.mean(axis=0)) if uniform else clean(E[li]))[idx]
                return probe(X, y)

            sel = at(pick)
            fin = at(final)
            uni = at(0, uniform=True)
            clu = at(clust_pick[tag])
            rows.append(dict(
                model=NAMES[tag], tag=tag, test_corpus=corpus,
                n=sel[2], n_classes=sel[3], n_layers=n_layers,
                clf_picked_layer=pick, clustering_picked_layer=clust_pick[tag],
                clf_sel_acc=sel[0], clf_sel_sd=sel[1],
                final_acc=fin[0], uniform_acc=uni[0],
                clustering_layer_acc=clu[0],
                gain_vs_final=sel[0] - fin[0],
                gain_vs_uniform=sel[0] - uni[0],
                cost_of_using_clustering_layer=sel[0] - clu[0]))
            print(f"      {corpus:12s} sel {sel[0]:.3f}  final {fin[0]:.3f}  "
                  f"uniform {uni[0]:.3f}  clustering-layer {clu[0]:.3f}  "
                  f"(n={sel[2]}, {sel[3]} speakers)", flush=True)
        del emb, z, oemb, zo

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/T_layer_transfer_classification.tsv")
    df.to_csv(out, sep="\t", index=False)

    print("\n## T. Layer transfer for speaker CLASSIFICATION\n")
    for c in ("OpenSLR-30", "Omnilingual"):
        d = df[df.test_corpus == c]
        print(f"  --- test = {c} ({d.n.iloc[0]} utts, {d.n_classes.iloc[0]} speakers) ---")
        print(d[["model", "clf_picked_layer", "clustering_picked_layer",
                 "clf_sel_acc", "final_acc", "uniform_acc",
                 "clustering_layer_acc", "gain_vs_final",
                 "cost_of_using_clustering_layer"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"    beats final: {int((d.gain_vs_final > 0).sum())}/{len(d)} "
              f"median {d.gain_vs_final.median():+.3f}")
        print(f"    beats all-layer mean: {int((d.gain_vs_uniform > 0).sum())}/{len(d)} "
              f"median {d.gain_vs_uniform.median():+.3f}")
        print(f"    cost of reusing the clustering layer: median "
              f"{d.cost_of_using_clustering_layer.median():+.3f}, "
              f"max {d.cost_of_using_clustering_layer.max():+.3f}\n")
    agree = (df.clf_picked_layer == df.clustering_picked_layer).sum()
    print(f"  layer agreement between the two task types: {agree}/{len(df)} rows")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
