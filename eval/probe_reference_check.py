#!/usr/bin/env python3
"""A discriminating comparison of the GPU probe against scikit-learn.

The first comparison used a 12-speaker task where both implementations reached
1.000, which cannot distinguish them. This uses a representation where the
probe actually struggles - a FINAL-layer embedding, which the layer study shows
is the weakest for speaker identity - on a speaker subset small enough for
scikit-learn to handle.

Accuracy alone is still a coarse test, so this also compares the training loss
each implementation reaches and the predicted probability vectors themselves.
Two solvers of the same convex objective should agree closely on both.

Writes eval/analysis/S12_probe_reference_hard.tsv.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SPEAKERS = 60          # small enough for sklearn, large enough to be hard


def fit_gpu(Xtr, ytr, Xte, k, l2=1.0, steps=200):
    Xt = torch.as_tensor(Xtr, device=DEV)
    yt = torch.as_tensor(ytr, device=DEV, dtype=torch.long)
    Xv = torch.as_tensor(Xte, device=DEV)
    W = torch.zeros(Xt.shape[1], k, device=DEV, requires_grad=True)
    b = torch.zeros(k, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, history_size=10,
                            line_search_fn="strong_wolfe")
    lf = torch.nn.CrossEntropyLoss()

    def cl():
        opt.zero_grad()
        loss = lf(Xt @ W + b, yt) + l2 / (2 * len(Xt)) * (W * W).sum()
        loss.backward()
        return loss

    opt.step(cl)
    with torch.no_grad():
        # training loss WITHOUT the penalty, so it matches sklearn's log loss
        train_ce = float(lf(Xt @ W + b, yt))
        proba = torch.softmax(Xv @ W + b, dim=1).cpu().numpy()
    return train_ce, proba


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str}).set_index("utt_id")
    rows = []
    for tag, layer, label in [("hubert", -1, "final"), ("whisper", -1, "final"),
                              ("hubert", 3, "selected")]:
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        utt = z["utt"].astype(str)
        meta = sub.loc[utt]
        i52 = np.flatnonzero(meta.dataset.values == "openslr52")
        spk = meta.speaker_id.values[i52]
        keep_spk = set(sorted(set(spk))[:N_SPEAKERS])
        sel = i52[np.isin(spk, list(keep_spk))]
        X = np.nan_to_num(z["emb"][layer][sel].astype(np.float32))
        y = meta.speaker_id.values[sel]
        codes, uniq = pd.factorize(y)

        Xtr, Xte, ytr, yte = train_test_split(X, codes, test_size=0.4,
                                              random_state=0, stratify=codes)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = ((Xtr - mu) / sd).astype(np.float32)
        Xte = ((Xte - mu) / sd).astype(np.float32)

        g_loss, g_p = fit_gpu(Xtr, ytr, Xte, len(uniq))
        sk = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        s_p = sk.predict_proba(Xte)
        from sklearn.metrics import log_loss
        s_loss = log_loss(ytr, sk.predict_proba(Xtr), labels=np.arange(len(uniq)))

        g_acc = float((g_p.argmax(1) == yte).mean())
        s_acc = float((s_p.argmax(1) == yte).mean())
        agree = float((g_p.argmax(1) == s_p.argmax(1)).mean())
        rows.append(dict(
            model=tag, layer=label, n_classes=len(uniq), n_train=len(ytr),
            gpu_acc=g_acc, sklearn_acc=s_acc, acc_diff=g_acc - s_acc,
            gpu_train_ce=g_loss, sklearn_train_ce=s_loss,
            ce_diff=g_loss - s_loss,
            pred_agreement=agree,
            max_abs_proba_diff=float(np.abs(g_p - s_p).max()),
            mean_abs_proba_diff=float(np.abs(g_p - s_p).mean())))
        print(f"  {tag}/{label:8s} {len(uniq)} spk | acc {g_acc:.4f} vs {s_acc:.4f} "
              f"| train CE {g_loss:.4f} vs {s_loss:.4f} | same prediction "
              f"{agree:.3f} | max |Δp| {np.abs(g_p-s_p).max():.4f}", flush=True)
        del z

    d = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/S12_probe_reference_hard.tsv")
    d.to_csv(out, sep="\t", index=False)
    print(f"\n  worst accuracy gap   : {d.acc_diff.abs().max():.4f}")
    print(f"  worst train-CE gap   : {d.ce_diff.abs().max():.4f}")
    print(f"  lowest prediction agreement: {d.pred_agreement.min():.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
