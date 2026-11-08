#!/usr/bin/env python3
"""MAEB (arXiv:2602.16008) evaluation protocol, instantiated on Sinhala speech.

The paper evaluates a frozen audio encoder by mean-pooling the final transformer
layer over time and then running a fixed set of task-type evaluators on top of
those vectors. Nothing is fine-tuned. This script reproduces the five evaluators
that our label structure supports, over the corpora in data/manifest.tsv:

  Classification     few-shot logistic probe, 8 examples per class   -> accuracy
  Clustering         MiniBatchKMeans, k = number of true labels      -> v_measure
  PairClassification cosine similarity of an audio pair              -> max_ap
  Retrieval          cosine ranking over an audio corpus             -> cv_recall_at_5
  Reranking          cosine ranking of pre-selected hard candidates  -> map_at_1000

Model ranking uses the Borda count over tasks, as in MAEB/MMTEB, alongside the
mean score.

Tasks are built from two label structures already present in the subset:
  speaker_id     - who is speaking (an acoustic/paralinguistic label)
  content_group  - utterances sharing a transcript, read by DIFFERENT speakers
                   (a linguistic label; the speaker-invariant one)
MAEB's headline finding is that acoustic and linguistic ability dissociate, and
this pairing is what lets us test that on a single low-resource language.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (v_measure_score, adjusted_rand_score,
                             average_precision_score)

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"


# ---------------------------------------------------------------- utilities
def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def sim_matrix(X, Y, device, chunk=2048):
    """Cosine similarity of L2-normalised X [n,D] against Y [m,D], on the GPU."""
    Xt = torch.as_tensor(X, device=device)
    Yt = torch.as_tensor(Y, device=device)
    out = torch.empty(len(X), len(Y), device=device)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = Xt[s:s + chunk] @ Yt.T
    return out


# ------------------------------------------------------------- evaluators
def classification(X, y, rng, n_shot=8, n_exp=10, max_test=4096):
    """MAEB: 'a logistic regression is trained on audio embeddings', few-shot
    linear probing with 8 examples per class. We repeat the draw n_exp times and
    average, since with 8 shots a single draw is high variance."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    classes = classes[counts >= n_shot + 1]          # need at least one held-out
    keep = np.isin(y, classes)
    X, y = X[keep], y[keep]
    if len(classes) < 2:
        return float("nan")

    by_class = {c: np.flatnonzero(y == c) for c in classes}
    accs = []
    for _ in range(n_exp):
        tr = np.concatenate([rng.choice(ix, n_shot, replace=False)
                             for ix in by_class.values()])
        te = np.setdiff1d(np.arange(len(y)), tr, assume_unique=False)
        if len(te) > max_test:
            te = rng.choice(te, max_test, replace=False)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def clustering(X, y, rng, n_exp=5, seed=0):
    """MAEB: 'MiniBatchKMeans (with k set to the number of true labels) and
    V-measure'.

    Two deviations from a naive implementation, both forced by this data:

    1. We resample whole LABEL GROUPS, never individual points. Subsampling
       points shatters small classes, and V-measure is not chance-corrected, so
       a partition into many near-singleton clusters scores ~0.92 by homogeneity
       alone. MAEB's own clustering tasks have few classes with many members
       each, so the protocol never meets this regime; we have to construct it.
    2. We report ARI alongside, and a random-partition baseline, so a score can
       be read against what shuffling achieves on the same label structure.
    """
    y = np.asarray(y)
    k_all = len(np.unique(y))
    if k_all < 2:
        return {"v_measure": float("nan"), "ari": float("nan"),
                "rand_v_measure": float("nan"), "k": 0, "n": 0}
    vs, ars, rvs, rars = [], [], [], []
    for e in range(n_exp):
        km = MiniBatchKMeans(n_clusters=k_all, random_state=seed + e, n_init=3,
                             batch_size=1024, max_iter=100).fit(X)
        pred = km.labels_
        vs.append(v_measure_score(y, pred))
        ars.append(adjusted_rand_score(y, pred))
        r = rng.integers(0, k_all, size=len(y))
        rvs.append(v_measure_score(y, r))
        rars.append(adjusted_rand_score(y, r))
    return {"v_measure": float(np.mean(vs)), "ari": float(np.mean(ars)),
            "rand_v_measure": float(np.mean(rvs)),
            "rand_ari": float(np.mean(rars)), "k": k_all, "n": len(y)}


def pair_classification(X, pairs, labels, device):
    """MAEB: 'similarity is computed between embeddings, and average precision
    based on cosine similarity serves as the main metric'."""
    a, b = pairs[:, 0], pairs[:, 1]
    Xt = torch.as_tensor(X, device=device)
    sims = (Xt[a] * Xt[b]).sum(1).float().cpu().numpy()
    return float(average_precision_score(labels, sims))


def retrieval(X, qidx, rel, device, exclude=None, ks=(1, 5, 10), corpus=None):
    """MAEB: 'documents are ranked by cosine similarity, with CV Recall@5'.

    rel[i] is the boolean row of relevant corpus documents for query qidx[i].
    A query counts as a hit at k if ANY relevant document is in its top k.
    `exclude` masks candidates that must not be ranked (the query itself, and,
    for the content task, the query speaker's own utterances).
    """
    C = X if corpus is None else X[corpus]
    S = sim_matrix(X[qidx], C, device)
    if exclude is not None:
        S = S.masked_fill(torch.as_tensor(exclude, device=device), -2.0)
    R = torch.as_tensor(rel, device=device)
    valid = R.any(1)
    S, R = S[valid], R[valid]
    order = S.argsort(1, descending=True)
    hits = torch.gather(R, 1, order).float()
    out = {f"cv_recall_at_{k}": float(hits[:, :k].amax(1).mean()) for k in ks}
    # nDCG@10 with binary gains
    disc = 1.0 / torch.log2(torch.arange(2, 12, device=device).float())
    dcg = (hits[:, :10] * disc).sum(1)
    npos = R.sum(1).clamp(max=10).long()
    ideal = torch.stack([disc[:n].sum() for n in range(11)])[npos]
    out["ndcg_at_10"] = float((dcg / ideal.clamp(min=1e-9)).mean())
    out["n_queries"] = int(valid.sum())
    return out


def reranking(X, queries, device):
    """MAEB: 'reranking evaluates ranking quality on pre-selected candidate sets
    containing relevant documents and hard negatives', metric MAP@1000."""
    aps = []
    Xt = torch.as_tensor(X, device=device)
    for q, cands, labs in queries:
        sims = (Xt[cands] @ Xt[q]).float()
        order = sims.argsort(descending=True)[:1000]
        rel = torch.as_tensor(labs, device=device, dtype=torch.float32)[order]
        if rel.sum() == 0:
            continue
        ranks = torch.arange(1, len(rel) + 1, device=device, dtype=torch.float32)
        prec = rel.cumsum(0) / ranks
        aps.append(float((prec * rel).sum() / rel.sum()))
    return float(np.mean(aps)) if aps else float("nan")


# ------------------------------------------------------------- task builders
def build_tasks(sub, rng, n_pairs=20000, n_rerank=2000, n_hard_neg=100,
                n_spk_clusters=50, n_content_clusters=150, min_spk_utts=9):
    """Build one set of tasks PER DATASET, plus the one cross-corpus task.

    MAEB treats every dataset as its own task, and here that is also forced by
    the data: openslr30 is 48 kHz studio audio from 12 speakers, openslr52 is
    16 kHz from 476. Pooling them lets a model separate "different speaker" pairs
    by recording channel alone, which inflates every speaker task. So each task
    is built inside a single corpus, and the only cross-corpus task is the one
    whose label IS the corpus.
    """
    ds_all = sub.dataset.values
    spk_all = pd.factorize(sub.speaker_id.values)[0]
    grp_raw = sub.content_group.values
    in_grp_all = grp_raw != ""

    tasks = {}
    skipped = []

    # --- the one legitimately cross-corpus task ----------------------------
    tasks["Corpus_Classification"] = dict(
        kind="classification", dataset="openslr30+52", idx=np.arange(len(ds_all)),
        labels=(ds_all == "openslr52").astype(int))

    for ds in sorted(set(ds_all)):
        pool = np.flatnonzero(ds_all == ds)
        tag = {"openslr30": "OpenSLR30", "openslr52": "OpenSLR52"}.get(ds, ds)
        spk = spk_all[pool]                    # speaker codes, dataset-local
        spk = pd.factorize(spk)[0]
        gl = grp_raw[pool]
        in_grp = in_grp_all[pool]
        grp = np.full(len(pool), -1)
        if in_grp.any():
            grp[in_grp] = pd.factorize(gl[in_grp])[0]
        n = len(pool)

        # ---- Classification: speaker identity within the corpus
        sz = pd.Series(spk).value_counts()
        if (sz >= min_spk_utts).sum() >= 2:
            tasks[f"{tag}_SpeakerClassification"] = dict(
                kind="classification", dataset=ds, idx=pool, labels=spk)
        else:
            skipped.append((f"{tag}_SpeakerClassification", "fewer than 2 usable speakers"))

        # ---- Clustering: k = number of speakers kept (whole classes only)
        elig = sz[sz >= 20].index.to_numpy()
        if len(elig) >= 2:
            if len(elig) > n_spk_clusters:
                elig = rng.choice(elig, n_spk_clusters, replace=False)
            sel = np.isin(spk, elig)
            tasks[f"{tag}_SpeakerClustering"] = dict(
                kind="clustering", dataset=ds, idx=pool[sel], labels=spk[sel])
        else:
            skipped.append((f"{tag}_SpeakerClustering", "fewer than 2 speakers with >=20 utts"))

        # ---- PairClassification: same speaker vs different speaker
        by_spk = pd.Series(np.arange(n)).groupby(spk).apply(lambda x: x.values)
        ids = [i for i in by_spk.index if len(by_spk[i]) >= 2]
        if len(ids) >= 2:
            npair = min(n_pairs, n * 4) // 2
            pos, neg = [], []
            while len(pos) < npair:
                i = ids[rng.integers(len(ids))]
                a, b = rng.choice(by_spk[i], 2, replace=False)
                pos.append((pool[a], pool[b]))
            while len(neg) < npair:
                a, b = rng.integers(n, size=2)
                if spk[a] != spk[b]:
                    neg.append((pool[a], pool[b]))
            tasks[f"{tag}_SpeakerPairClassification"] = dict(
                kind="pair", dataset=ds, pairs=np.array(pos + neg),
                labels=np.r_[np.ones(len(pos)), np.zeros(len(neg))])

        # ---- Retrieval: find another utterance by the same speaker
        q = rng.choice(n, min(2000, n), replace=False)
        rel = (spk[q][:, None] == spk[None, :])
        excl = np.zeros_like(rel)
        excl[np.arange(len(q)), q] = True
        rel = rel & ~excl
        if rel.any():
            tasks[f"{tag}_SpeakerA2ARetrieval"] = dict(
                kind="retrieval", dataset=ds, qidx=pool[q], rel=rel,
                exclude=excl, corpus=pool)

        # ---- Content tasks: only where the corpus actually has cross-speaker
        #      transcript repeats. openslr30 has 26 such utterances in 13 groups,
        #      which is far too few, so its content tasks are skipped outright.
        gi = np.flatnonzero(in_grp)
        gsz = pd.Series(grp[gi]).value_counts()
        if len(gi) < 200 or (gsz >= 2).sum() < 50:
            skipped.append((f"{tag}_Content*",
                            f"only {len(gi)} utts in {len(gsz)} content groups"))
            continue

        big = gsz[gsz >= 3].index.to_numpy()
        if len(big) >= 2:
            if len(big) > n_content_clusters:
                big = rng.choice(big, n_content_clusters, replace=False)
            sel = np.isin(grp, big)
            tasks[f"{tag}_ContentClustering"] = dict(
                kind="clustering", dataset=ds, idx=pool[sel], labels=grp[sel])

        by_grp = pd.Series(gi).groupby(grp[gi]).apply(lambda x: x.values)
        gids = [g for g in by_grp.index if len(by_grp[g]) >= 2]
        npair = min(n_pairs, len(gi) * 2) // 2
        pos, neg = [], []
        while len(pos) < npair:
            g = gids[rng.integers(len(gids))]
            a, b = rng.choice(by_grp[g], 2, replace=False)
            if spk[a] != spk[b]:
                pos.append((pool[a], pool[b]))
        while len(neg) < npair:
            a, b = rng.choice(gi, 2, replace=False)
            if grp[a] != grp[b] and spk[a] != spk[b]:
                neg.append((pool[a], pool[b]))
        tasks[f"{tag}_ContentPairClassification"] = dict(
            kind="pair", dataset=ds, pairs=np.array(pos + neg),
            labels=np.r_[np.ones(len(pos)), np.zeros(len(neg))])

        # Retrieval / reranking: same sentence, DIFFERENT voice. Same-speaker
        # candidates are removed so voice matching cannot score.
        qc = rng.choice(gi, min(2000, len(gi)), replace=False)
        same = spk[qc][:, None] == spk[None, :]
        exc = same.copy()
        exc[np.arange(len(qc)), qc] = True
        relc = (grp[qc][:, None] == grp[None, :]) & (grp[qc][:, None] >= 0) & ~exc
        tasks[f"{tag}_ContentA2ARetrieval"] = dict(
            kind="retrieval", dataset=ds, qidx=pool[qc], rel=relc,
            exclude=exc, corpus=pool)

        qr = rng.choice(gi, min(n_rerank, len(gi)), replace=False)
        rrq = []
        for q_ in qr:
            posc = np.flatnonzero((grp == grp[q_]) & (spk != spk[q_]))
            if len(posc) == 0:
                continue
            hard = np.flatnonzero((spk == spk[q_]) & (np.arange(n) != q_))
            if len(hard) > n_hard_neg:
                hard = rng.choice(hard, n_hard_neg, replace=False)
            easy = rng.choice(n, n_hard_neg, replace=False)
            cands = np.unique(np.r_[posc, hard, easy])
            cands = cands[cands != q_]
            labs = np.isin(cands, posc).astype(np.float32)
            if labs.sum() and labs.sum() < len(labs):
                rrq.append((int(pool[q_]), pool[cands], labs))
        if rrq:
            tasks[f"{tag}_ContentReranking"] = dict(
                kind="reranking", dataset=ds, queries=rrq)

    if skipped:
        print("  skipped tasks (insufficient data):", flush=True)
        for nm_, why in skipped:
            print(f"    {nm_:38s} {why}", flush=True)
    return tasks


MAIN_METRIC = {
    "classification": "accuracy", "clustering": "v_measure", "pair": "max_ap",
    "retrieval": "cv_recall_at_5", "reranking": "map_at_1000",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "eval/emb"))
    ap.add_argument("--subset", default=os.path.join(ROOT, "eval/subset.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/maeb_results.tsv"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "eval/maeb_summary.tsv"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--layer", default="last",
                    help="'last' (the MAEB protocol), 'all', or an integer index")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sub = pd.read_csv(a.subset, sep="\t", quoting=3, dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    meta = sub.set_index("utt_id")
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    tags = a.models or sorted(f[:-4] for f in os.listdir(a.emb) if f.endswith(".npz"))
    rows = []

    for tag in tags:
        p = os.path.join(a.emb, f"{tag}.npz")
        if not os.path.exists(p):
            print(f"[{tag}] missing, skipping", file=sys.stderr); continue
        z = np.load(p, allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        L = emb.shape[0]
        # tasks are built once per model, against that model's own utterance order
        m = meta.loc[utt].reset_index()
        rng = np.random.default_rng(a.seed)
        tasks = build_tasks(m, rng)

        layers = (list(range(L)) if a.layer == "all"
                  else [L - 1] if a.layer == "last" else [int(a.layer)])
        print(f"\n=== {tag}: {emb.shape} (layers, utts, dim) | layers {layers}",
              flush=True)

        for li in layers:
            Xall = emb[li].astype(np.float32)
            nbad = int((~np.isfinite(Xall)).any(1).sum())
            if nbad:
                Xall = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
            Xall = l2norm(Xall)     # cosine geometry throughout, as in MAEB

            for name, t in tasks.items():
                t0 = time.time()
                rng_t = np.random.default_rng(a.seed)
                k = t["kind"]
                if k == "classification":
                    sc = {"accuracy": classification(Xall[t["idx"]], t["labels"], rng_t)}
                elif k == "clustering":
                    sc = clustering(Xall[t["idx"]], t["labels"], rng_t, seed=a.seed)
                elif k == "pair":
                    sc = {"max_ap": pair_classification(Xall, t["pairs"], t["labels"], dev)}
                elif k == "retrieval":
                    sc = retrieval(Xall, t["qidx"], t["rel"], dev, t["exclude"],
                                   corpus=t.get("corpus"))
                elif k == "reranking":
                    sc = {"map_at_1000": reranking(Xall, t["queries"], dev)}
                main = sc[MAIN_METRIC[k]]
                rows.append(dict(model=tag, layer=li, task=name,
                                 dataset=t.get("dataset", ""), task_type=k,
                                 main_metric=MAIN_METRIC[k], score=main,
                                 n_nonfinite=nbad,
                                 **{f"m_{kk}": vv for kk, vv in sc.items()}))
                print(f"  [L{li}] {name:38s} {MAIN_METRIC[k]:>15s} = {main:.4f}"
                      f"   ({time.time()-t0:.1f}s)", flush=True)
        del emb, z

    df = pd.DataFrame(rows)
    cols = ["model", "dataset", "task", "task_type", "layer", "main_metric", "score"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(a.out, sep="\t", index=False, float_format="%.6f")
    print(f"\nwrote {a.out}  ({len(df)} rows)")

    # ---- Borda count over tasks, as in MAEB/MMTEB --------------------------
    # each task is a voter ranking the models; a model gets (n_models - rank)
    # points per task. Reported next to the mean, which MAEB also publishes.
    best = df.loc[df.groupby(["model", "task"]).score.idxmax()]
    piv = best.pivot(index="model", columns="task", values="score")
    nm = len(piv)
    borda = piv.rank(axis=0, ascending=False, method="average").rsub(nm + 1).sum(1)
    summ = pd.DataFrame({"mean_score": piv.mean(1), "borda_points": borda})
    for tt, d in best.groupby("task_type"):
        summ[f"mean_{tt}"] = d.pivot(index="model", columns="task",
                                     values="score").mean(1)
    summ = summ.sort_values("borda_points", ascending=False)
    summ.insert(0, "maeb_rank", np.arange(1, len(summ) + 1))
    summ.to_csv(a.summary, sep="\t", float_format="%.4f")

    pd.set_option("display.width", 200)
    print("\n--- per-task scores (best layer per model) ---")
    print(piv.round(4).to_string())
    print("\n--- MAEB-style ranking ---")
    print(summ.round(4).to_string())
    print(f"\nwrote {a.summary}")


if __name__ == "__main__":
    main()
