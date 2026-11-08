#!/usr/bin/env python3
"""Convergence for every representation the classification table reports.

An earlier check covered only the selected layer for three encoders. The
reported table also contains the final layer and the all-layer mean, and a
probe can fail to converge on one representation while succeeding on another -
the final layer is exactly where the problem is hardest, since the features
carry least speaker information there.

For each encoder and each of the three reported conditions: initial loss, final
loss, gradient norm, iterations used, and whether the iteration cap was hit.

Writes eval/analysis/S15_convergence_all_conditions.tsv.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def fit(X, y, k, l2=1.0, steps=200):
    Xt = torch.as_tensor(X, device=DEV)
    yt = torch.as_tensor(y, device=DEV, dtype=torch.long)
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

    init = float(cl())
    opt.step(cl)
    with torch.no_grad():
        final = float(lf(Xt @ W + b, yt) + l2 / (2 * len(Xt)) * (W * W).sum())
    g = torch.cat([W.grad.flatten(), b.grad.flatten()]).norm().item()
    it = int(opt.state[opt._params[0]].get("n_iter", -1))
    del Xt, yt, W, b
    return init, final, g, it


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str}).set_index("utt_id")
    U = pd.read_csv(os.path.join(ROOT, "eval/analysis/U_classification_476_speakers.tsv"),
                    sep="\t").set_index("model")
    rows = []
    for tag in sorted(NAMES):
        name = NAMES[tag]
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]
        i52 = np.flatnonzero(meta.dataset.values == "openslr52")
        rng = np.random.default_rng(0)
        sco = rng.permutation(i52)[len(i52) // 2:]
        y_all = meta.speaker_id.values[sco]
        picked = int(U.loc[name, "clf_picked_layer"])
        final_l = emb.shape[0] - 1

        for cond, X in (("selected", emb[picked][sco]),
                        ("final", emb[final_l][sco]),
                        ("all_layer_mean", emb.mean(axis=0)[sco])):
            X = np.nan_to_num(X.astype(np.float32))
            keep = pd.Series(y_all).groupby(y_all).transform("size").values >= 4
            Xk, yk = X[keep], np.asarray(y_all)[keep]
            codes, uniq = pd.factorize(yk)
            Xtr, _, ytr, _ = train_test_split(Xk, codes, test_size=0.4,
                                              random_state=0, stratify=codes)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            init, fin, g, it = fit(((Xtr - mu) / sd).astype(np.float32), ytr, len(uniq))
            rows.append(dict(model=name, condition=cond,
                             layer=picked if cond == "selected" else
                                   (final_l if cond == "final" else -1),
                             n_classes=len(uniq), init_loss=init, final_loss=fin,
                             grad_norm=g, n_iter=it, hit_cap=it >= 200))
            print(f"  {name:32s} {cond:15s} loss {init:.3f}->{fin:.4f} "
                  f"grad {g:.1e} iters {it}/200 cap {it>=200}", flush=True)
        del emb, z

    d = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/S15_convergence_all_conditions.tsv")
    d.to_csv(out, sep="\t", index=False)
    print(f"\n  conditions checked: {len(d)}")
    print(f"  any hit the iteration cap: {bool(d.hit_cap.any())}")
    if d.hit_cap.any():
        print(d[d.hit_cap][["model", "condition", "n_iter", "grad_norm"]].to_string(index=False))
    print(f"  worst gradient norm: {d.grad_norm.max():.2e}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
