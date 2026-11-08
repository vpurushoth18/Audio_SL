#!/usr/bin/env python3
"""Export the ACTUAL classification partitions, then re-check 200 vs 2,000 on them.

A first version of this script created `np.random.default_rng(0)` inside the
model loop, so every model received the same permutation. The original
experiment (eval/classification_476.py) creates the generator ONCE before the
loop and calls `rng.permutation` inside it, so the generator state advances and
each model gets a different split. The exported manifest was therefore wrong
for five of six models.

This reproduces the original exactly:

    rng = np.random.default_rng(0)          # once, before the loop
    for tag in sorted(NAMES):               # same model order
        half = rng.permutation(i52)         # state advances per model

Model order, embedding row order and the >=4-example filter are unchanged.

The reproduction is then CHECKED rather than assumed: mean accuracy at a
200-iteration cap must match U_classification_476_speakers.tsv within
tolerance, and the script reports the comparison. If it does not match, that
has to be resolved before the iteration check means anything.

Outputs:
  classification_splits.tsv            partition and per-seed fold membership
  classification_iteration_check.tsv   200 vs 2000 on those exact partitions
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2)
MIN_PER_CLASS = 4
TEST_FRAC = 0.4
TOL = 5e-3                    # tolerance for the reproduction check

# the same mapping, in the same insertion order, as classification_476.py
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def fit(Xtr, ytr, Xte, yte, k, steps, l2=1.0):
    """The probe exactly as the reported experiment fits it."""
    Xt = torch.as_tensor(Xtr, device=DEV)
    yt = torch.as_tensor(ytr, device=DEV, dtype=torch.long)
    Xv = torch.as_tensor(Xte, device=DEV)
    yv = torch.as_tensor(yte, device=DEV, dtype=torch.long)
    W = torch.zeros(Xt.shape[1], k, device=DEV, requires_grad=True)
    b = torch.zeros(k, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, history_size=10,
                            line_search_fn="strong_wolfe")
    lf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lf(Xt @ W + b, yt) + l2 / (2 * len(Xt)) * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = float(((Xv @ W + b).argmax(1) == yv).float().mean())
        fin = float(lf(Xt @ W + b, yt) + l2 / (2 * len(Xt)) * (W * W).sum())
    g = torch.cat([W.grad.flatten(), b.grad.flatten()]).norm().item()
    it = int(opt.state[opt._params[0]].get("n_iter", -1))
    del Xt, yt, Xv, yv, W, b
    return acc, fin, g, it


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str}).set_index("utt_id")
    U = pd.read_csv(os.path.join(ROOT, "eval/analysis/U_classification_476_speakers.tsv"),
                    sep="\t").set_index("model")

    rng = np.random.default_rng(0)          # ONCE, before the loop
    split_rows, iter_rows, repro = [], [], []

    for tag in sorted(NAMES):               # same order as the original
        name = NAMES[tag]
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]
        i52 = np.flatnonzero(meta.dataset.values == "openslr52")

        half = rng.permutation(i52)         # state advances per model
        parts = {"selection": half[:len(half) // 2], "scoring": half[len(half) // 2:]}
        picked, last = int(U.loc[name, "clf_picked_layer"]), emb.shape[0] - 1

        # ---------- split manifest
        for part, idx in parts.items():
            y = meta.speaker_id.values[idx]
            keep = pd.Series(y).groupby(y).transform("size").values >= MIN_PER_CLASS
            kept_pos = np.flatnonzero(keep)
            codes = pd.factorize(np.asarray(y)[keep])[0]
            folds = {s: {} for s in SEEDS}
            if kept_pos.size:
                for s in SEEDS:
                    tr, te = train_test_split(np.arange(len(kept_pos)),
                                              test_size=TEST_FRAC, random_state=s,
                                              stratify=codes)
                    for i in tr:
                        folds[s][int(kept_pos[i])] = "train"
                    for i in te:
                        folds[s][int(kept_pos[i])] = "test"
            for j, (u, sp, k) in enumerate(zip(utt[idx], y, keep)):
                split_rows.append(dict(
                    model=name, utt_id=u, speaker_id=sp, partition=part,
                    kept_after_min4_filter=bool(k),
                    **{f"fold_seed{s}": folds[s].get(j, "excluded") for s in SEEDS}))

        # ---------- iteration check on the SCORING half the table reports
        idx = parts["scoring"]
        y = meta.speaker_id.values[idx]
        keep = pd.Series(y).groupby(y).transform("size").values >= MIN_PER_CLASS
        for cond, li, X0 in (("selected", picked, emb[picked][idx]),
                             ("final", last, emb[last][idx]),
                             ("all_layer_mean", -1, emb.mean(axis=0)[idx])):
            X = np.nan_to_num(X0.astype(np.float32))[keep]
            yy = np.asarray(y)[keep]
            codes, uniq = pd.factorize(yy)
            for s in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(X, codes, test_size=TEST_FRAC,
                                                      random_state=s, stratify=codes)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                A = ((Xtr - mu) / sd).astype(np.float32)
                B = ((Xte - mu) / sd).astype(np.float32)
                a2, l2_, g2, i2 = fit(A, ytr, B, yte, len(uniq), 200)
                a20, l20, g20, i20 = fit(A, ytr, B, yte, len(uniq), 2000)
                iter_rows.append(dict(
                    model=name, condition=cond, layer=li, seed=s,
                    n_train=len(ytr), n_test=len(yte), n_classes=len(uniq),
                    acc_cap200=a2, acc_cap2000=a20, acc_delta=a20 - a2,
                    iters_cap200=i2, iters_cap2000=i20,
                    cap200_reached=i2 >= 200, converged_by_2000=i20 < 2000,
                    final_loss_cap200=l2_, final_loss_cap2000=l20,
                    grad_norm_cap200=g2, grad_norm_cap2000=g20))

            mine = np.mean([r["acc_cap200"] for r in iter_rows
                            if r["model"] == name and r["condition"] == cond])
            col = {"selected": "sel_acc", "final": "final_acc",
                   "all_layer_mean": "uniform_acc"}[cond]
            orig = float(U.loc[name, col])
            repro.append(dict(model=name, condition=cond, original=orig,
                              reproduced=float(mine), diff=float(mine - orig),
                              within_tol=bool(abs(mine - orig) <= TOL)))
            print(f"  {name:32s} {cond:15s} orig {orig:.4f}  repro {mine:.4f}  "
                  f"diff {mine-orig:+.5f}  {'OK' if abs(mine-orig)<=TOL else 'MISMATCH'}",
                  flush=True)
        del emb, z

    sp = pd.DataFrame(split_rows)
    sp.to_csv(os.path.join(ROOT, "eval/analysis/classification_splits.tsv"),
              sep="\t", index=False)
    it = pd.DataFrame(iter_rows)
    it.to_csv(os.path.join(ROOT, "eval/analysis/classification_iteration_check.tsv"),
              sep="\t", index=False, float_format="%.17g")
    rp = pd.DataFrame(repro)
    rp.to_csv(os.path.join(ROOT, "eval/analysis/classification_reproduction_check.tsv"),
              sep="\t", index=False, float_format="%.17g")

    print("\n=== reproduction against U_classification_476_speakers.tsv ===")
    print(rp.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\n  within tolerance ({TOL}): {int(rp.within_tol.sum())}/{len(rp)}")
    print(f"  largest absolute difference: {rp['diff'].abs().max():.5f}")

    # splits must differ between models now; identical halves would mean the
    # generator was reset and the reproduction is wrong again
    first = sp[(sp.model == NAMES["hubert"]) & (sp.partition == "scoring")].utt_id
    same = [m for m in sp.model.unique() if m != NAMES["hubert"] and
            set(sp[(sp.model == m) & (sp.partition == "scoring")].utt_id) == set(first)]
    print(f"\n  models sharing hubert's scoring half: {len(same)} "
          f"(expected 0 - the generator advances per model)")

    print(f"\n  split rows: {len(sp):,}")
    print(f"  dropped by the >=4 filter: "
          f"{int((~sp.kept_after_min4_filter).sum())} rows across all models")
    print(f"  iteration-check rows: {len(it)}")
    print(f"  reached the 200 cap: {int(it.cap200_reached.sum())}/{len(it)}")
    print(f"  all converged by 2000: {bool(it.converged_by_2000.all())}")
    print(f"  largest accuracy change 200->2000: {it.acc_delta.abs().max():.5f}")


if __name__ == "__main__":
    main()
