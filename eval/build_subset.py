#!/usr/bin/env python3
"""Stratified evaluation subset for intrinsic embedding evaluation.

Two label structures are needed and they pull in opposite directions:
  * content  - groups of utterances sharing a transcript, read by DIFFERENT speakers
  * speaker  - speakers with enough utterances to be separable
So we sample content groups first, then top up speakers that came out thin.
"""
import argparse, os
import pandas as pd, numpy as np

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=4000, help="cross-speaker content groups")
    ap.add_argument("--min-spk-utts", type=int, default=30)
    ap.add_argument("--topup", type=int, default=20, help="min utts per speaker after top-up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "subset.tsv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    mf = pd.read_csv(os.path.join(ROOT, "data/manifest.tsv"), sep="\t",
                     quoting=3, dtype={"speaker_id": str})
    mf["norm"] = (mf.transcript.fillna("").astype(str)
                    .str.replace(r"\s+", " ", regex=True).str.strip())

    keep = []

    # ---- OpenSLR-52: content groups + speaker balance
    s52 = mf[(mf.dataset == "openslr52") & (mf.has_transcript == 1)].copy()
    g = s52.groupby("norm").agg(n=("utt_id", "size"), nspk=("speaker_id", "nunique"))
    xs = g[(g.n > 1) & (g.nspk > 1)].index.to_numpy()
    pick = rng.choice(xs, size=min(a.groups, len(xs)), replace=False)
    content = s52[s52.norm.isin(pick)].copy()
    content["content_group"] = content.norm
    keep.append(content)

    # top up speakers that are under-represented in the content sample
    have = content.speaker_id.value_counts()
    elig = s52.speaker_id.value_counts()
    elig = elig[elig >= a.min_spk_utts].index
    extra = []
    for spk in elig:
        need = a.topup - int(have.get(spk, 0))
        if need <= 0:
            continue
        pool = s52[(s52.speaker_id == spk) & (~s52.utt_id.isin(content.utt_id))]
        if len(pool):
            extra.append(pool.sample(n=min(need, len(pool)), random_state=a.seed))
    if extra:
        ex = pd.concat(extra); ex["content_group"] = ""
        keep.append(ex)

    # ---- OpenSLR-30: take all of it (small, and the clean 48 kHz contrast)
    s30 = mf[mf.dataset == "openslr30"].copy()
    dup30 = s30[s30.has_transcript == 1].norm
    dup30 = set(dup30[dup30.duplicated(keep=False)])
    s30["content_group"] = np.where(s30.norm.isin(dup30), s30.norm, "")
    keep.append(s30)

    sub = pd.concat(keep).drop_duplicates(subset="utt_id").reset_index(drop=True)

    # a content group is only usable if it still has >1 speaker after sampling
    gs = sub[sub.content_group != ""].groupby("content_group").speaker_id.nunique()
    usable = set(gs[gs > 1].index)
    sub["content_group"] = np.where(sub.content_group.isin(usable), sub.content_group, "")

    sub = sub.drop(columns=["norm"])
    sub.to_csv(a.out, sep="\t", index=False)

    print(f"subset: {len(sub):,} utts | {sub.duration_s.sum()/3600:.2f} h")
    for ds, d in sub.groupby("dataset"):
        cg = d[d.content_group != ""]
        print(f"  [{ds}] n={len(d):,} spk={d.speaker_id.nunique()} "
              f"content_groups={cg.content_group.nunique():,} in_groups={len(cg):,}")
    spk = sub.speaker_id.value_counts()
    print(f"  speakers>=20 utts: {(spk>=20).sum()} / {len(spk)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
