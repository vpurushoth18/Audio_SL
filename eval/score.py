#!/usr/bin/env python3
"""Intrinsic scoring of the pooled embeddings, per model x dataset x layer.

The subset carries two label structures that pull in opposite directions, so we
score both and report the gap:

  speaker  - 1-NN speaker accuracy (cosine, leave-one-out). High = the layer
             encodes WHO is speaking.
  content  - retrieval over content groups (utterances sharing a transcript),
             with same-speaker candidates REMOVED from the ranking. Without that
             exclusion a model scores well just by matching the voice, since a
             speaker's own repeats sit closest. High = the layer encodes WHAT
             was said, independently of the voice.

Reported per layer:
  spk_top1      1-NN speaker accuracy
  cnt_r1        cross-speaker content recall@1
  cnt_map       cross-speaker content mean average precision
  gap           cnt_map - spk_top1 (content-vs-speaker selectivity)
  spk_sil       silhouette of speaker labels (cosine)

Writes eval/scores.tsv (long form, one row per model/dataset/layer).
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"


def l2norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def score_layer(X, spk, grp, device, chunk=1024):
    """X [N,D] float32 on device, already L2-normalised.

    spk: int64 [N] speaker ids. grp: int64 [N] content-group ids, -1 = not in a group.
    """
    N = X.shape[0]
    idx = torch.arange(N, device=device)

    spk_hits = 0
    # content metrics are averaged over queries that are in a group AND have at
    # least one cross-speaker positive available
    ap_sum, r1_sum, n_q = 0.0, 0.0, 0
    npos_sum, ncand_sum = 0.0, 0.0

    # silhouette accumulators (cosine distance = 1 - sim)
    sil_a = torch.zeros(N, device=device)
    sil_b = torch.full((N,), float("inf"), device=device)
    uspk = torch.unique(spk)
    onehot = (spk[:, None] == uspk[None, :]).float()          # [N, K]
    cnts = onehot.sum(0)                                      # [K]

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = X[s:e] @ X.T                                   # [c, N]
        rows = idx[s:e]
        self_mask = rows[:, None] == idx[None, :]
        sim_ns = sim.masked_fill(self_mask, -2.0)

        # --- speaker 1-NN
        nn = sim_ns.argmax(1)
        spk_hits += (spk[nn] == spk[rows]).sum().item()

        # --- speaker silhouette: mean intra vs best mean inter, cosine distance
        # one-hot matmul gives all per-cluster distance sums in a single kernel
        dist = 1.0 - sim                                      # [c, N]
        tot = dist @ onehot                                   # [c, K] sums per cluster
        own = (spk[rows][:, None] == uspk[None, :])           # [c, K]
        # the query's own row contributes 0 distance to its own cluster, but must
        # be dropped from the count, otherwise a singleton cluster divides by 0
        denom = torch.where(own, (cnts - 1).clamp(min=1), cnts).float()
        means = tot / denom
        sil_a[s:e] = means.masked_fill(~own, 0.0).sum(1)
        sil_b[s:e] = means.masked_fill(own, float("inf")).min(1).values

        # --- content retrieval, same-speaker candidates removed
        q_in = grp[rows] >= 0
        if q_in.any():
            qsim = sim_ns[q_in]                               # [q, N]
            qrows = rows[q_in]
            same_spk = spk[qrows][:, None] == spk[None, :]
            qsim = qsim.masked_fill(same_spk, -2.0)           # cross-speaker only
            pos = (grp[qrows][:, None] == grp[None, :]) & (~same_spk)
            npos = pos.sum(1)
            ok = npos > 0
            if ok.any():
                qsim, pos, npos = qsim[ok], pos[ok], npos[ok]
                order = qsim.argsort(dim=1, descending=True)
                pos_sorted = torch.gather(pos, 1, order).float()
                r1_sum += pos_sorted[:, 0].sum().item()
                csum = pos_sorted.cumsum(1)
                ranks = torch.arange(1, pos_sorted.shape[1] + 1, device=device).float()
                prec = csum / ranks
                ap = (prec * pos_sorted).sum(1) / npos.float()
                ap_sum += ap.sum().item()
                n_q += int(ok.sum().item())
                # random-ranking baseline: a query's positives over the
                # cross-speaker candidates it is actually ranked against
                ncand = (qsim > -2.0).sum(1)
                npos_sum += npos.float().sum().item()
                ncand_sum += ncand.float().sum().item()

    sil = ((sil_b - sil_a) / torch.maximum(sil_a, sil_b).clamp(min=1e-8)).mean().item()
    return dict(
        spk_top1=spk_hits / N,
        cnt_r1=(r1_sum / n_q) if n_q else float("nan"),
        cnt_map=(ap_sum / n_q) if n_q else float("nan"),
        spk_sil=sil,
        rnd_map=(npos_sum / ncand_sum) if ncand_sum else float("nan"),
        n_utts=N,
        n_content_queries=n_q,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "eval/emb"))
    ap.add_argument("--subset", default=os.path.join(ROOT, "eval/subset.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/scores.tsv"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sub = pd.read_csv(a.subset, sep="\t", quoting=3, dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    meta = sub.set_index("utt_id")

    tags = a.models or [f[:-4] for f in sorted(os.listdir(a.emb)) if f.endswith(".npz")]
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    rows = []

    for tag in tags:
        p = os.path.join(a.emb, f"{tag}.npz")
        if not os.path.exists(p):
            print(f"[{tag}] missing {p}, skipping", file=sys.stderr); continue
        z = np.load(p, allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        L = emb.shape[0]
        m = meta.loc[utt]
        print(f"\n=== {tag}: {emb.shape} (layers, utts, dim)", flush=True)

        for ds, dmask in m.groupby("dataset").groups.items():
            sel = np.isin(utt, np.asarray(dmask))
            spk_codes = pd.factorize(m.speaker_id.values[sel])[0]
            g = m.content_group.values[sel]
            grp_codes = pd.factorize(np.where(g == "", None, g))[0]
            grp_codes = np.where(g == "", -1, grp_codes)
            spk = torch.as_tensor(spk_codes, device=dev, dtype=torch.int64)
            grp = torch.as_tensor(grp_codes, device=dev, dtype=torch.int64)
            t0 = time.time()
            for li in range(L):
                raw = emb[li][sel].astype(np.float32)
                # a layer that overflowed during the forward pass must not be
                # scored as if it were valid - count it and zero the bad rows
                nbad = int((~np.isfinite(raw)).any(1).sum())
                if nbad:
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                X = torch.as_tensor(raw, device=dev, dtype=torch.float32)
                r = score_layer(l2norm(X), spk, grp, dev)
                r["n_nonfinite"] = nbad
                r.update(model=tag, dataset=ds, layer=li)
                r["gap"] = r["cnt_map"] - r["spk_top1"]
                rows.append(r)
            print(f"  [{ds}] {L} layers in {(time.time()-t0)/60:.1f} min", flush=True)
        del emb, z

    df = pd.DataFrame(rows)[
        ["model", "dataset", "layer", "spk_top1", "spk_sil", "cnt_r1", "cnt_map",
         "rnd_map", "gap", "n_utts", "n_content_queries", "n_nonfinite"]
    ]
    df.to_csv(a.out, sep="\t", index=False, float_format="%.6f")
    print(f"\nwrote {a.out}  ({len(df)} rows)")
    for (mdl, ds), d in df.groupby(["model", "dataset"]):
        b = d.loc[d.cnt_map.idxmax()]
        print(f"  {mdl:10s} [{ds}] best content layer {int(b.layer):2d}  "
              f"cnt_map={b.cnt_map:.3f} spk_top1={b.spk_top1:.3f}")


if __name__ == "__main__":
    main()
