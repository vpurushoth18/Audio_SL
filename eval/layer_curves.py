#!/usr/bin/env python3
"""Save the FULL per-layer curves, for both capabilities, so they can be plotted.

The earlier sweeps computed a score at every layer and then kept only three
points: the selected layer, the final layer and the all-layer average. That is
enough to state the result but not to draw it, and the layer curve is the figure
that carries the main finding.

Three curves per encoder, every layer:

  speaker_clf      476-speaker classification accuracy on the OpenSLR-52
                   SELECTION half, GPU probe
  speaker_clust    speaker clustering AMI on OpenSLR-30
  content_pref     P[content match closer than speaker match] on OpenSLR-52

These are DEVELOPMENT curves: they are what a practitioner would look at to
choose a layer, and they are computed on the same selection half that the
held-out tables choose their layer on. The held-out scores live in
U_classification_476_speakers.tsv and are computed on the disjoint scoring
half, so the two are deliberately different numbers for different purposes and
must not be quoted interchangeably.

Depth is reported normalised, so architectures of different depths overlay.

Writes eval/analysis/V_layer_curves.tsv.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def clean(a):
    return np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def probe_gpu(X, y, seeds=(0, 1), l2_pen=1.0):
    """Multinomial logistic regression on the GPU. Classes need >=4 examples."""
    keep = pd.Series(y).groupby(y).transform("size").values >= 4
    X, y = X[keep], np.asarray(y)[keep]
    codes, uniq = pd.factorize(y)
    accs = []
    for s in seeds:
        Xa, Xb, ya, yb = train_test_split(X, codes, test_size=0.4,
                                          random_state=s, stratify=codes)
        # fitted on train only; see layer_transfer_clf.probe
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
        Xa = ((Xa - mu) / sd).astype(np.float32)
        Xb = ((Xb - mu) / sd).astype(np.float32)
        Xa_t = torch.as_tensor(Xa, device=DEVICE)
        Xb_t = torch.as_tensor(Xb, device=DEVICE)
        ya_t = torch.as_tensor(ya, device=DEVICE, dtype=torch.long)
        yb_t = torch.as_tensor(yb, device=DEVICE, dtype=torch.long)
        W = torch.zeros(Xa.shape[1], len(uniq), device=DEVICE, requires_grad=True)
        b = torch.zeros(len(uniq), device=DEVICE, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=150, history_size=10,
                                line_search_fn="strong_wolfe")
        lossf = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = (lossf(Xa_t @ W + b, ya_t)
                    + l2_pen / (2 * len(Xa_t)) * (W * W).sum())
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            accs.append(((Xb_t @ W + b).argmax(1) == yb_t).float().mean().item())
        del Xa_t, Xb_t, ya_t, yb_t, W, b
    return float(np.mean(accs))


def ami(X, y, seeds=(0, 1, 2)):
    yy = pd.factorize(y)[0]
    k = len(np.unique(yy))
    out = [adjusted_mutual_info_score(
        yy, MiniBatchKMeans(n_clusters=k, random_state=s, n_init=5,
                            batch_size=512).fit_predict(l2(X))) for s in seeds]
    return float(np.mean(out))


def content_pref(E, q, c, sm):
    A = l2(E)
    a = (A[q] * A[c]).sum(1)
    b = (A[q] * A[sm]).sum(1)
    w = (a > b).astype(float)
    w[np.isclose(a, b, rtol=0, atol=1e-6)] = 0.5
    return float(w.mean())


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    sub = sub.set_index("utt_id")

    rows = []
    for tag in sorted(NAMES):
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]
        i52_all = np.flatnonzero(meta.dataset.values == "openslr52")
        # the SAME selection/scoring halves the held-out table uses, so the
        # development curve and the held-out score are not computed on
        # overlapping utterances
        split_rng = np.random.default_rng(0)
        half = split_rng.permutation(i52_all)
        i52 = half[:len(half) // 2]          # selection half
        i30 = np.flatnonzero(meta.dataset.values == "openslr30")
        n_layers = emb.shape[0]
        final = n_layers - 1

        # triplets once, reused for every layer
        rng = np.random.default_rng(0)
        spk, grp = meta.speaker_id.values, meta.content_group.values
        by_g, by_s = {}, {}
        for i, (g, sp) in enumerate(zip(grp, spk)):
            if g:
                by_g.setdefault(g, []).append(i)
            by_s.setdefault(sp, []).append(i)
        q, c, sm = [], [], []
        for i in range(len(utt)):
            g, sp = grp[i], spk[i]
            if not g:
                continue
            cc = [j for j in by_g[g] if j != i and spk[j] != sp]
            cs = [j for j in by_s[sp] if j != i and grp[j] != g]
            if cc and cs:
                q.append(i)
                c.append(int(rng.choice(cc)))
                sm.append(int(rng.choice(cs)))
        q, c, sm = np.array(q), np.array(c), np.array(sm)

        for li in range(n_layers):
            E = clean(emb[li])
            rows.append(dict(
                model=NAMES[tag], tag=tag, layer=li, n_layers=n_layers,
                depth=li / max(final, 1),
                speaker_clf=probe_gpu(E[i52], meta.speaker_id.values[i52]),
                speaker_clust=ami(E[i30], meta.speaker_id.values[i30]),
                content_pref=content_pref(E, q, c, sm)))
        last = rows[-1]
        best_clf = max(r["speaker_clf"] for r in rows if r["tag"] == tag)
        best_cnt = max(r["content_pref"] for r in rows if r["tag"] == tag)
        print(f"  {NAMES[tag]:34s} {n_layers} layers | peak speaker_clf "
              f"{best_clf:.3f} | peak content_pref {best_cnt:.3f} | "
              f"final {last['speaker_clf']:.3f}/{last['content_pref']:.3f}",
              flush=True)
        del emb, z

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/V_layer_curves.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"\n{len(df)} rows across {df.tag.nunique()} encoders")
    print("\nlayer of peak score, by capability:")
    for tag, g in df.groupby("tag"):
        a = g.loc[g.speaker_clf.idxmax()]
        b = g.loc[g.content_pref.idxmax()]
        print(f"  {NAMES[tag]:34s} speaker L{int(a.layer)} (depth {a.depth:.2f}) | "
              f"content L{int(b.layer)} (depth {b.depth:.2f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
