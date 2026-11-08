#!/usr/bin/env python3
"""How well STRUCTURED is each embedding space? Per model x dataset x layer.

This asks a different question from retrieval accuracy: ignoring any task, does
the embedding space fall into clean clusters on its own, and do those clusters
correspond to something real (speaker identity)?

Three families of measure:

  1. Is there cluster structure at all?  (no labels used)
       mean_cos   mean pairwise cosine. Near 1.0 means every vector points the
                  same way - the "narrow cone" collapse. Such a space has no
                  usable structure no matter what the labels say.
       eff_rank   participation ratio of the PCA spectrum: how many dimensions
                  actually carry variance, out of dim. Low = collapsed.
       sil_km     silhouette of k-means labels. High = the space really does
                  separate into k blobs.

  2. Do the natural clusters match the speakers?  (k-means vs true labels)
       ari, nmi   agreement between the k-means partition and speaker identity.
       purity     fraction of points whose k-means cluster is dominated by
                  their own speaker.

  3. How good are the speaker clusters, taken as given?  (true labels)
       sil_spk    silhouette under cosine distance, -1..1
       dbi        Davies-Bouldin, lower is better
       chi        Calinski-Harabasz, higher is better
       cnt_sil    silhouette of the content groups, over in-group utterances
                  only - the same geometry question asked of "what was said"

Writes eval/cluster_scores.tsv (long form).
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd, torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             davies_bouldin_score, calinski_harabasz_score)

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"


def l2norm_np(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


@torch.no_grad()
def silhouette_cosine(X, labels, device, chunk=2048):
    """Exact silhouette under cosine distance, computed on the GPU.

    X [N,D] L2-normalised torch tensor; labels int64 [N].
    Clusters of size 1 get silhouette 0, the standard convention.
    """
    N = X.shape[0]
    uniq = torch.unique(labels)
    if uniq.numel() < 2:
        return float("nan")
    onehot = (labels[:, None] == uniq[None, :]).float()
    cnts = onehot.sum(0)
    a = torch.zeros(N, device=device)
    b = torch.full((N,), float("inf"), device=device)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dist = 1.0 - (X[s:e] @ X.T)
        tot = dist @ onehot
        own = labels[s:e][:, None] == uniq[None, :]
        denom = torch.where(own, (cnts - 1).clamp(min=1), cnts).float()
        means = tot / denom
        a[s:e] = means.masked_fill(~own, 0.0).sum(1)
        b[s:e] = means.masked_fill(own, float("inf")).min(1).values

    singleton = (cnts[torch.searchsorted(uniq, labels)] <= 1)
    sil = (b - a) / torch.maximum(a, b).clamp(min=1e-8)
    sil = torch.where(singleton, torch.zeros_like(sil), sil)
    return sil.mean().item()


@torch.no_grad()
def mean_pairwise_cos(X, chunk=2048):
    """Mean off-diagonal cosine similarity - the anisotropy of the space."""
    N = X.shape[0]
    tot = 0.0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        tot += (X[s:e] @ X.T).sum().item()
    return (tot - N) / (N * (N - 1))


def eff_rank(x):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2.

    Equals dim for a perfectly isotropic space, 1 if all variance is on one axis.
    """
    xc = x - x.mean(0, keepdims=True)
    # eigenvalues of the covariance via the Gram matrix of whichever side is smaller
    if xc.shape[0] >= xc.shape[1]:
        c = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    else:
        c = (xc @ xc.T) / max(xc.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(c.astype(np.float64))
    ev = np.clip(ev, 0, None)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def purity(true, pred):
    df = pd.crosstab(pred, true)
    return float(df.max(axis=1).sum() / df.values.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "eval/emb"))
    ap.add_argument("--subset", default=os.path.join(ROOT, "eval/subset.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/cluster_scores.tsv"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sub = pd.read_csv(a.subset, sep="\t", quoting=3, dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    meta = sub.set_index("utt_id")

    tags = a.models or sorted(f[:-4] for f in os.listdir(a.emb) if f.endswith(".npz"))
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    rows = []

    for tag in tags:
        p = os.path.join(a.emb, f"{tag}.npz")
        if not os.path.exists(p):
            print(f"[{tag}] missing, skipping", file=sys.stderr); continue
        z = np.load(p, allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        L, _, D = emb.shape
        m = meta.loc[utt]
        print(f"\n=== {tag}: {emb.shape}", flush=True)

        for ds in sorted(m.dataset.unique()):
            sel = (m.dataset.values == ds)
            spk = pd.factorize(m.speaker_id.values[sel])[0]
            g = m.content_group.values[sel]
            in_grp = g != ""
            grp = pd.factorize(g[in_grp])[0]
            k = len(np.unique(spk))
            t0 = time.time()

            for li in range(L):
                x = emb[li][sel].astype(np.float32)
                nbad = int((~np.isfinite(x)).any(1).sum())
                if nbad:
                    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                xn = l2norm_np(x)
                X = torch.as_tensor(xn, device=dev)

                km = MiniBatchKMeans(n_clusters=k, random_state=a.seed, n_init=3,
                                     batch_size=1024, max_iter=100).fit(xn)
                pred = km.labels_

                r = dict(
                    model=tag, dataset=ds, layer=li, dim=D, k_clusters=k,
                    mean_cos=mean_pairwise_cos(X),
                    eff_rank=eff_rank(xn),
                    sil_km=silhouette_cosine(X, torch.as_tensor(pred, device=dev,
                                                                dtype=torch.int64), dev),
                    ari=adjusted_rand_score(spk, pred),
                    nmi=normalized_mutual_info_score(spk, pred),
                    purity=purity(spk, pred),
                    sil_spk=silhouette_cosine(X, torch.as_tensor(spk, device=dev,
                                                                 dtype=torch.int64), dev),
                    dbi=davies_bouldin_score(xn, spk),
                    chi=calinski_harabasz_score(xn, spk),
                    cnt_sil=silhouette_cosine(
                        torch.as_tensor(l2norm_np(x[in_grp]), device=dev),
                        torch.as_tensor(grp, device=dev, dtype=torch.int64), dev)
                        if in_grp.sum() > 2 else float("nan"),
                    n_utts=int(sel.sum()), n_in_groups=int(in_grp.sum()),
                    n_nonfinite=nbad,
                )
                rows.append(r)
            print(f"  [{ds}] k={k}  {L} layers in {(time.time()-t0)/60:.1f} min", flush=True)
        del emb, z

    df = pd.DataFrame(rows)[
        ["model", "dataset", "layer", "dim", "k_clusters",
         "mean_cos", "eff_rank", "sil_km", "ari", "nmi", "purity",
         "sil_spk", "dbi", "chi", "cnt_sil",
         "n_utts", "n_in_groups", "n_nonfinite"]
    ]
    df.to_csv(a.out, sep="\t", index=False, float_format="%.6f")
    print(f"\nwrote {a.out}  ({len(df)} rows)")
    for (mdl, ds), d in df.groupby(["model", "dataset"]):
        b = d.loc[d.nmi.idxmax()]
        print(f"  {mdl:10s} [{ds}] best speaker-cluster layer {int(b.layer):2d}  "
              f"NMI={b.nmi:.3f} ARI={b.ari:.3f} purity={b.purity:.3f} sil_spk={b.sil_spk:.3f}")


if __name__ == "__main__":
    main()
