#!/usr/bin/env python3
"""Does a layer chosen on development data still win on held-out data?

The layer sweep reports the best layer per model, but it chooses that layer on
the same data it reports. That is a biased comparison, so it cannot support a
recommendation. This script does the honest version:

  1. choose the layer on a DEV split,
  2. freeze it,
  3. evaluate on a disjoint TEST split,
  4. compare against the default final layer.

Two splits, because the two capabilities have different data constraints:

  speaker   dev = OpenSLR-52, test = OpenSLR-30. A genuine cross-corpus
            transfer: different speakers, different recording conditions,
            different sample rate at source.
  content   OpenSLR-30 has only 13 usable content groups, far too few, so the
            content split is WITHIN OpenSLR-52 with disjoint content groups AND
            disjoint speakers between dev and test. That is held-out data, not
            a second corpus, and is reported as such.

Metrics are the chance-corrected ones:

  speaker   AMI of k-means against speaker labels (V-measure inflates with
            many clusters; on this data it overstated by 0.245)
  content   P[sim(q, same words other speaker) > sim(q, same speaker other
            words)], the controlled diagnostic

Writes eval/analysis/J_layer_transfer.tsv.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
RNG = np.random.default_rng(0)

MODEL_NAMES = {
    "xlsr300m": "facebook/wav2vec2-xls-r-300m",
    "mms300m": "facebook/mms-300m",
    "w2vbert2": "facebook/w2v-bert-2.0",
    "whisper": "openai/whisper-large-v3",
    "wavlm": "microsoft/wavlm-large",
    "hubert": "facebook/hubert-large-ls960-ft",
}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def speaker_ami(E, labels, seed=0):
    y = pd.factorize(labels)[0]
    k = len(np.unique(y))
    if k < 2 or len(y) < k:
        return np.nan
    pred = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                           batch_size=1024).fit_predict(l2(E))
    return float(adjusted_mutual_info_score(y, pred))


def content_pref(E, tri):
    """P[content match closer than speaker match], ties split."""
    if len(tri) == 0:
        return np.nan
    A = l2(E)
    sc = (A[tri.q.values] * A[tri.content.values]).sum(1)
    ss = (A[tri.q.values] * A[tri.speaker.values]).sum(1)
    tie = np.isclose(sc, ss, rtol=0, atol=1e-6)
    win = (sc > ss).astype(float)
    win[tie] = 0.5
    return float(win.mean())


def make_triplets(meta, idx_pool, seed=0):
    """Triplets restricted to rows in idx_pool (positions into the full array)."""
    rng = np.random.default_rng(seed)
    pool = set(idx_pool.tolist())
    spk = meta.speaker_id.values
    grp = meta.content_group.values

    by_group, by_spk = {}, {}
    for i in idx_pool:
        if grp[i]:
            by_group.setdefault(grp[i], []).append(i)
        by_spk.setdefault(spk[i], []).append(i)

    rows = []
    for i in idx_pool:
        g, s = grp[i], spk[i]
        if not g:
            continue
        cc = [j for j in by_group.get(g, []) if j != i and spk[j] != s]
        cs = [j for j in by_spk.get(s, []) if j != i and grp[j] != g]
        if not cc or not cs:
            continue
        rows.append((i, int(rng.choice(cc)), int(rng.choice(cs))))
    return pd.DataFrame(rows, columns=["q", "content", "speaker"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "eval/emb"))
    ap.add_argument("--subset", default=os.path.join(ROOT, "eval/subset.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/analysis"))
    ap.add_argument("--dev-frac", type=float, default=0.5,
                    help="fraction of OpenSLR-52 speakers used as dev for the "
                         "content split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sub = pd.read_csv(a.subset, sep="\t", quoting=3, dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    sub = sub.set_index("utt_id")

    rows = []
    for tag in sorted(f[:-4] for f in os.listdir(a.emb) if f.endswith(".npz")):
        z = np.load(os.path.join(a.emb, f"{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]
        n_layers = emb.shape[0]

        is52 = (meta.dataset.values == "openslr52")
        is30 = (meta.dataset.values == "openslr30")

        # ---- speaker: dev = OpenSLR-52, test = OpenSLR-30 (cross-corpus)
        spk_dev = np.flatnonzero(is52)
        spk_test = np.flatnonzero(is30)

        # ---- content: within OpenSLR-52, disjoint speakers AND groups
        rng = np.random.default_rng(a.seed)
        spk52 = np.array(sorted(set(meta.speaker_id.values[is52])))
        rng.shuffle(spk52)
        cut = int(len(spk52) * a.dev_frac)
        dev_spk, test_spk = set(spk52[:cut]), set(spk52[cut:])
        dev_mask = is52 & np.isin(meta.speaker_id.values, list(dev_spk))
        test_mask = is52 & np.isin(meta.speaker_id.values, list(test_spk))
        # drop groups that straddle the split, so no transcript is in both
        g_dev = set(meta.content_group.values[dev_mask]) - {""}
        g_test = set(meta.content_group.values[test_mask]) - {""}
        shared = g_dev & g_test
        dev_mask &= ~np.isin(meta.content_group.values, list(shared))
        test_mask &= ~np.isin(meta.content_group.values, list(shared))

        tri_dev = make_triplets(meta, np.flatnonzero(dev_mask), a.seed)
        tri_test = make_triplets(meta, np.flatnonzero(test_mask), a.seed)

        # ---- sweep layers on dev only
        spk_dev_scores, cnt_dev_scores = [], []
        for li in range(n_layers):
            E = np.nan_to_num(emb[li].astype(np.float32), nan=0.0,
                              posinf=0.0, neginf=0.0)
            spk_dev_scores.append(speaker_ami(E[spk_dev],
                                              meta.speaker_id.values[spk_dev]))
            cnt_dev_scores.append(content_pref(E, tri_dev))

        spk_pick = int(np.nanargmax(spk_dev_scores))
        cnt_pick = int(np.nanargmax(cnt_dev_scores))
        final = n_layers - 1

        # ---- evaluate the frozen choices, and the default, on test
        def at(li, which):
            E = np.nan_to_num(emb[li].astype(np.float32), nan=0.0,
                              posinf=0.0, neginf=0.0)
            if which == "speaker":
                return speaker_ami(E[spk_test],
                                   meta.speaker_id.values[spk_test])
            return content_pref(E, tri_test)

        rows.append(dict(
            model=MODEL_NAMES.get(tag, tag), tag=tag, n_layers=n_layers,
            capability="speaker",
            dev="OpenSLR-52", test="OpenSLR-30 (cross-corpus)",
            n_dev=int(spk_dev.size), n_test=int(spk_test.size),
            picked_layer=spk_pick, picked_depth=spk_pick / max(final, 1),
            dev_score=spk_dev_scores[spk_pick],
            test_score_picked=at(spk_pick, "speaker"),
            test_score_final=at(final, "speaker"),
        ))
        rows.append(dict(
            model=MODEL_NAMES.get(tag, tag), tag=tag, n_layers=n_layers,
            capability="content",
            dev="OpenSLR-52 speakers A", test="OpenSLR-52 speakers B (held-out)",
            n_dev=len(tri_dev), n_test=len(tri_test),
            picked_layer=cnt_pick, picked_depth=cnt_pick / max(final, 1),
            dev_score=cnt_dev_scores[cnt_pick],
            test_score_picked=at(cnt_pick, "content"),
            test_score_final=at(final, "content"),
        ))
        del emb, z

    df = pd.DataFrame(rows)
    df["gain_over_final"] = df.test_score_picked - df.test_score_final
    df = df.sort_values(["capability", "gain_over_final"], ascending=[True, False])
    df.to_csv(os.path.join(a.out, "J_layer_transfer.tsv"), sep="\t", index=False)

    for cap in ("speaker", "content"):
        d = df[df.capability == cap]
        print(f"\n## Layer transfer: {cap}")
        print(f"   dev = {d.dev.iloc[0]} -> test = {d.test.iloc[0]}")
        print(f"   metric = {'AMI vs speaker labels' if cap=='speaker' else 'P[content>speaker]'}\n")
        print(d[["model", "n_layers", "picked_layer", "picked_depth",
                 "dev_score", "test_score_picked", "test_score_final",
                 "gain_over_final"]].to_string(index=False))

    print("\n  picked_layer is chosen on DEV only, then frozen. gain_over_final\n"
          "  is the honest benefit of layer selection: positive means a\n"
          "  dev-chosen intermediate layer beats the default final layer on\n"
          "  held-out data. Negative means layer selection did not transfer.")
    print(f"\nwrote {a.out}/J_layer_transfer.tsv")


if __name__ == "__main__":
    main()
