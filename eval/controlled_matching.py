#!/usr/bin/env python3
"""Controlled speaker-vs-content matching, and the layer study.

The capability profiles say Whisper is strong on content retrieval and
WavLM/XLS-R on speaker tasks. Those are aggregate results over tasks that
differ in corpus, metric and candidate construction, so they are suggestive
rather than controlled. This isolates the comparison.

For a query utterance q, from the same corpus, three candidates:

    content match   same transcript, DIFFERENT speaker
    speaker match   DIFFERENT transcript, same speaker
    plain negative  different transcript, different speaker

The diagnostic is

    P[ sim(q, content match) > sim(q, speaker match) ]

A model above 0.5 puts linguistic content ahead of voice; below 0.5 it puts
voice first. The complementary speaker framing is the same comparison read the
other way, so both are reported from one experiment.

Three things that decide whether this is evidence or decoration:

* TIES. Reported explicitly and split half-half, never silently dropped.
* THE RESAMPLING UNIT. Queries from one speaker are not independent, so the
  bootstrap resamples SPEAKERS (a cluster bootstrap), not queries. Resampling
  queries would understate the CI several-fold.
* DURATION. Longer audio gives a longer mean-pooled average and can shift
  similarity on its own, so candidates are optionally duration-matched and the
  unmatched result is reported alongside.

Because the cached embeddings hold every layer, the same construction answers
the layer question: at which depth does each model prefer content over voice?

    python eval/controlled_matching.py                     # final layer, all models
    python eval/controlled_matching.py --layers all        # full layer sweep
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_triplets(meta, max_dur_ratio, seed):
    """One triplet per eligible query: (q, content_match, speaker_match, neg).

    Indices are positions into the embedding array.
    """
    rng = np.random.default_rng(seed)
    pos = {u: i for i, u in enumerate(meta.index)}
    spk = meta.speaker_id.values
    grp = meta.content_group.values
    dur = meta.duration_s.values

    by_group, by_spk = {}, {}
    for i, (g, s) in enumerate(zip(grp, spk)):
        if g:
            by_group.setdefault(g, []).append(i)
        by_spk.setdefault(s, []).append(i)

    rows = []
    for i in range(len(meta)):
        g, s = grp[i], spk[i]
        if not g:
            continue
        # content match: same transcript, different speaker
        cands_c = [j for j in by_group[g] if j != i and spk[j] != s]
        # speaker match: same speaker, different transcript
        cands_s = [j for j in by_spk[s] if j != i and grp[j] != g]
        if not cands_c or not cands_s:
            continue

        if max_dur_ratio:
            # keep candidates whose duration is within a ratio of the query's,
            # so neither candidate wins on length alone
            def near(js):
                return [j for j in js
                        if max(dur[j], dur[i]) / max(min(dur[j], dur[i]), 1e-6)
                        <= max_dur_ratio] or js
            cands_c, cands_s = near(cands_c), near(cands_s)

        c = int(rng.choice(cands_c))
        sm = int(rng.choice(cands_s))
        # plain negative: different transcript and different speaker
        for _ in range(20):
            n = int(rng.integers(0, len(meta)))
            if grp[n] != g and spk[n] != s:
                break
        else:
            continue
        rows.append((i, c, sm, n, s))
    return pd.DataFrame(rows, columns=["q", "content", "speaker", "neg",
                                       "q_speaker"])


def cos(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return (a * b).sum(1)


def cluster_bootstrap(values, groups, n=10000, alpha=0.05):
    """CI for the mean of `values`, resampling GROUPS (speakers) not rows.

    Queries from one speaker share that speaker's voice, so they are not
    independent observations; resampling speakers respects that.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    out = np.empty(n)
    for b in range(n):
        pick = uniq[RNG.integers(0, len(uniq), len(uniq))]
        sel = np.concatenate([idx_by_g[g] for g in pick])
        out[b] = values[sel].mean()
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def score_layer(E, tri, boot):
    """Return the content-over-speaker preference for one layer."""
    sc = cos(E[tri.q.values], E[tri.content.values])
    ss = cos(E[tri.q.values], E[tri.speaker.values])
    sn = cos(E[tri.q.values], E[tri.neg.values])

    tie = np.isclose(sc, ss, rtol=0, atol=1e-6)
    win = (sc > ss).astype(float)
    win[tie] = 0.5                      # ties split, never dropped
    p = float(win.mean())

    res = dict(
        p_content_over_speaker=p,
        n_queries=len(tri), n_ties=int(tie.sum()),
        mean_sim_content=float(sc.mean()),
        mean_sim_speaker=float(ss.mean()),
        mean_sim_negative=float(sn.mean()),
        # a sanity check: both real candidates should beat a plain negative,
        # otherwise the embedding carries neither signal
        p_content_over_negative=float((sc > sn).mean()),
        p_speaker_over_negative=float((ss > sn).mean()),
    )
    if boot:
        lo, hi = cluster_bootstrap(win, tri.q_speaker.values)
        res["ci_lo"], res["ci_hi"] = lo, hi
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "eval/emb"))
    ap.add_argument("--subset", default=os.path.join(ROOT, "eval/subset.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/analysis"))
    ap.add_argument("--layers", default="final", choices=["final", "all"])
    ap.add_argument("--max-dur-ratio", type=float, default=1.5,
                    help="duration-match candidates within this ratio; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sub = pd.read_csv(a.subset, sep="\t", quoting=3, dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    sub = sub.set_index("utt_id")

    tags = sorted(f[:-4] for f in os.listdir(a.emb) if f.endswith(".npz"))
    rows, layer_rows = [], []
    tri_cache = {}

    for tag in tags:
        z = np.load(os.path.join(a.emb, f"{tag}.npz"), allow_pickle=True)
        emb, utt = z["emb"], z["utt"].astype(str)
        meta = sub.loc[utt]

        key = (len(utt), a.max_dur_ratio)
        if key not in tri_cache:
            tri_cache[key] = build_triplets(meta, a.max_dur_ratio, a.seed)
        tri = tri_cache[key]

        n_layers = emb.shape[0]
        layers = [n_layers - 1] if a.layers == "final" else range(n_layers)
        for li in layers:
            E = np.nan_to_num(emb[li].astype(np.float32), nan=0.0,
                              posinf=0.0, neginf=0.0)
            boot = (a.layers == "final") or li == n_layers - 1
            r = score_layer(E, tri, boot)
            r.update(model=MODEL_NAMES.get(tag, tag), tag=tag, layer=li,
                     n_layers=n_layers, depth=li / max(n_layers - 1, 1))
            (rows if li == n_layers - 1 else layer_rows).append(r)
            if a.layers == "all":
                layer_rows.append(r) if li != n_layers - 1 else None
        del emb, z

    cols = ["model", "layer", "n_layers", "p_content_over_speaker", "ci_lo",
            "ci_hi", "n_queries", "n_ties", "mean_sim_content",
            "mean_sim_speaker", "mean_sim_negative",
            "p_content_over_negative", "p_speaker_over_negative"]
    fin = pd.DataFrame(rows)
    fin = fin[[c for c in cols if c in fin.columns]].sort_values(
        "p_content_over_speaker", ascending=False)
    fin.to_csv(os.path.join(a.out, "H_controlled_matching_final.tsv"),
               sep="\t", index=False)

    print("\n## Controlled matching, FINAL layer")
    print(f"   duration-matched within {a.max_dur_ratio}x; "
          f"CI = cluster bootstrap over speakers\n")
    print(fin.to_string(index=False))
    print("\n  p_content_over_speaker > 0.5 means the model ranks a different\n"
          "  speaker saying the SAME words above the same speaker saying\n"
          "  DIFFERENT words. Below 0.5 means voice dominates.")
    print("  p_*_over_negative are sanity checks: both should exceed 0.5, else\n"
          "  the embedding carries neither signal.")

    if a.layers == "all":
        lay = pd.DataFrame(layer_rows)
        lay = lay[[c for c in ["model", "tag", "layer", "n_layers", "depth",
                               "p_content_over_speaker", "mean_sim_content",
                               "mean_sim_speaker"] if c in lay.columns]]
        lay.to_csv(os.path.join(a.out, "I_controlled_matching_layers.tsv"),
                   sep="\t", index=False)
        print("\n## Layer sweep: depth at which content overtakes voice\n")
        for tag, g in lay.groupby("tag"):
            g = g.sort_values("layer")
            best = g.loc[g.p_content_over_speaker.idxmax()]
            cross = g[g.p_content_over_speaker > 0.5]
            print(f"  {MODEL_NAMES.get(tag, tag):34s} peak p="
                  f"{best.p_content_over_speaker:.3f} at layer "
                  f"{int(best.layer)}/{int(best.n_layers)-1} "
                  f"(depth {best.depth:.2f})"
                  + (f", content wins at layers "
                     f"{int(cross.layer.min())}-{int(cross.layer.max())}"
                     if len(cross) else ", content never wins"))
        print(f"\nwrote {a.out}/I_controlled_matching_layers.tsv")

    print(f"wrote {a.out}/H_controlled_matching_final.tsv")


if __name__ == "__main__":
    main()
