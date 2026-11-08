#!/usr/bin/env python3
"""Does the 200-iteration cap understate the baselines?

Convergence logging showed the selected layer always converging inside the cap
while the final layer and all-layer mean always hit it. Under-optimised
baselines would understate their accuracy and inflate the reported gain from
layer selection, so this refits every condition with a much larger budget and
compares accuracy directly.

If accuracy is unchanged, the cap cost nothing and the reported gains stand.
If the baselines improve, the gains in the classification table are overstated
by that amount and must be corrected.

Writes eval/analysis/S16_cap_sensitivity.tsv.
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
SEEDS = (0, 1, 2)


def fit_acc(Xtr, ytr, Xte, yte, k, steps, l2=1.0):
    Xt = torch.as_tensor(Xtr, device=DEV); yt = torch.as_tensor(ytr, device=DEV, dtype=torch.long)
    Xv = torch.as_tensor(Xte, device=DEV); yv = torch.as_tensor(yte, device=DEV, dtype=torch.long)
    W = torch.zeros(Xt.shape[1], k, device=DEV, requires_grad=True)
    b = torch.zeros(k, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, history_size=10,
                            line_search_fn="strong_wolfe")
    lf = torch.nn.CrossEntropyLoss()

    def cl():
        opt.zero_grad()
        loss = lf(Xt @ W + b, yt) + l2 / (2 * len(Xt)) * (W * W).sum()
        loss.backward(); return loss

    opt.step(cl)
    with torch.no_grad():
        acc = float(((Xv @ W + b).argmax(1) == yv).float().mean())
    it = int(opt.state[opt._params[0]].get("n_iter", -1))
    del Xt, yt, Xv, yv, W, b
    return acc, it


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
        sco = np.random.default_rng(0).permutation(i52)[len(i52) // 2:]
        y_all = meta.speaker_id.values[sco]
        picked = int(U.loc[name, "clf_picked_layer"]); last = emb.shape[0] - 1

        for cond, Xraw in (("selected", emb[picked][sco]), ("final", emb[last][sco]),
                           ("all_layer_mean", emb.mean(axis=0)[sco])):
            X = np.nan_to_num(Xraw.astype(np.float32))
            keep = pd.Series(y_all).groupby(y_all).transform("size").values >= 4
            Xk, yk = X[keep], np.asarray(y_all)[keep]
            codes, uniq = pd.factorize(yk)
            a200, a2000, its = [], [], []
            for sd_ in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(Xk, codes, test_size=0.4,
                                                      random_state=sd_, stratify=codes)
                mu, s = Xtr.mean(0), Xtr.std(0) + 1e-9
                Xtr = ((Xtr - mu) / s).astype(np.float32)
                Xte = ((Xte - mu) / s).astype(np.float32)
                a, _ = fit_acc(Xtr, ytr, Xte, yte, len(uniq), 200); a200.append(a)
                a, it = fit_acc(Xtr, ytr, Xte, yte, len(uniq), 2000); a2000.append(a); its.append(it)
            rows.append(dict(model=name, condition=cond,
                             acc_cap200=float(np.mean(a200)),
                             acc_cap2000=float(np.mean(a2000)),
                             delta=float(np.mean(a2000) - np.mean(a200)),
                             iters_at_2000=int(np.mean(its)),
                             converged_by_2000=bool(np.mean(its) < 2000)))
            print(f"  {name:32s} {cond:15s} 200:{np.mean(a200):.4f}  "
                  f"2000:{np.mean(a2000):.4f}  delta {np.mean(a2000)-np.mean(a200):+.4f}  "
                  f"iters {int(np.mean(its))}", flush=True)
        del emb, z

    d = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/S16_cap_sensitivity.tsv")
    d.to_csv(out, sep="\t", index=False)
    print(f"\n  largest accuracy change: {d.delta.abs().max():.4f}")
    print(d.groupby("condition").delta.agg(["mean", "max"]).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
