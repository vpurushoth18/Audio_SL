#!/usr/bin/env python3
"""Export the exact classification partitions, then re-check 200 vs 2000 on them.

Two deliverables, both keyed to the partitions the reported table actually used
rather than to a reconstruction:

  classification_splits.tsv
      model, utt_id, speaker_id, selection|scoring half, and for each seed
      whether the utterance was in the probe's train or test fold, plus a flag
      for anything dropped by the >=4-example filter.

  classification_iteration_check.tsv
      per seed, per condition (selected / final / all-layer mean): accuracy at
      a 200-iteration cap and at 2000, iterations used, and whether the cap was
      reached.

The split depends only on the permutation seed and the embedding row order,
which is identical across encoders, so the halves are shared; the probe folds
depend on the per-seed stratified split and are recorded per model.
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
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def fit(Xtr, ytr, Xte, yte, k, steps, l2=1.0):
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
        final_loss = float(lf(Xt @ W + b, yt) + l2 / (2 * len(Xt)) * (W * W).sum())
    g = torch.cat([W.grad.flatten(), b.grad.flatten()]).norm().item()
    it = int(opt.state[opt._params[0]].get("n_iter", -1))
    del Xt, yt, Xv, yv, W, b
    return acc, final_loss, g, it


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str}).set_index("utt_id")
    U = pd.read_csv(os.path.join(ROOT, "eval/analysis/U_classification_476_speakers.tsv"),
                    sep="\t").set_index("model")

    split_rows, iter_rows = [], []
    for tag in sorted(NAMES):
        name = NAMES[tag]
        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]
        i52 = np.flatnonzero(meta.dataset.values == "openslr52")
        half = np.random.default_rng(0).permutation(i52)
        parts = {"selection": half[:len(half) // 2], "scoring": half[len(half) // 2:]}
        picked, last = int(U.loc[name, "clf_picked_layer"]), emb.shape[0] - 1

        # ---- split manifest
        for part, idx in parts.items():
            y = meta.speaker_id.values[idx]
            keep = pd.Series(y).groupby(y).transform("size").values >= MIN_PER_CLASS
            codes_all = np.full(len(idx), -1)
            codes, _ = pd.factorize(np.asarray(y)[keep])
            codes_all[keep] = codes
            folds = {s: {} for s in SEEDS}
            if keep.sum():
                kept_pos = np.flatnonzero(keep)
                for s in SEEDS:
                    tr, te = train_test_split(kept_pos, test_size=TEST_FRAC,
                                              random_state=s, stratify=codes_all[kept_pos])
                    for i in tr: folds[s][i] = "train"
                    for i in te: folds[s][i] = "test"
            for j, (u, sp, k) in enumerate(zip(utt[idx], y, keep)):
                split_rows.append(dict(
                    model=name, utt_id=u, speaker_id=sp, partition=part,
                    kept_after_min4_filter=bool(k),
                    **{f"fold_seed{s}": folds[s].get(j, "excluded") for s in SEEDS}))

        # ---- iteration check on the SCORING partition, the one the table reports
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
            m = [r for r in iter_rows if r["model"] == name and r["condition"] == cond]
            print(f"  {name:32s} {cond:15s} 200:{np.mean([r['acc_cap200'] for r in m]):.4f} "
                  f"2000:{np.mean([r['acc_cap2000'] for r in m]):.4f} "
                  f"cap hit {sum(r['cap200_reached'] for r in m)}/{len(m)}", flush=True)
        del emb, z

    sp = pd.DataFrame(split_rows)
    sp.to_csv(os.path.join(ROOT, "eval/analysis/classification_splits.tsv"),
              sep="\t", index=False)
    it = pd.DataFrame(iter_rows)
    it.to_csv(os.path.join(ROOT, "eval/analysis/classification_iteration_check.tsv"),
              sep="\t", index=False)
    print(f"\n  split rows: {len(sp):,}  ({sp.utt_id.nunique():,} utts x {sp.model.nunique()} models)")
    print(f"  excluded by the >=4 filter: "
          f"{int((~sp.kept_after_min4_filter).sum() / sp.model.nunique())} utts per model")
    print(f"  iteration-check rows: {len(it)}")
    print(f"  largest accuracy change 200->2000: {it.acc_delta.abs().max():.5f}")
    print(f"  conditions reaching the 200 cap: {int(it.cap200_reached.sum())}/{len(it)}")
    print(f"  all converged by 2000: {bool(it.converged_by_2000.all())}")


if __name__ == "__main__":
    main()
