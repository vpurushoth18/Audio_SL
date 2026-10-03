#!/usr/bin/env python3
"""Audit the extraction: does padding leak, and do the high cosines matter?

Two checklist items, answered separately.

PADDING. If the mask were wrong, short clips batched with long ones would be
dragged toward the pad vector, so their embeddings would depend on batch
composition. Two tests:

  R1  re-encode a sample one clip at a time (batch=1, no padding possible) and
      compare against the batched cached embedding. A correct mask makes these
      agree to numerical precision.
  R2  check whether cosine similarity tracks the duration GAP between a pair.
      Leaked padding would make clips of similar length look similar
      regardless of content or speaker.

ANISOTROPY. Mean cosine between unrelated utterances is 0.94-0.99, which makes
absolute values uninformative. The question is whether it also distorts the
ORDERING the study relies on. Tests:

  R3  mean cosine for random pairs, same speaker, and same content, raw and
      after removing the mean embedding, with the participation ratio of the
      covariance spectrum as a measure of how many directions are in use.
  R4  recompute the controlled-matching diagnostic on mean-centred embeddings.
      If the preference order survives centring, anisotropy is cosmetic.

Writes eval/analysis/R_pooling_audit.tsv.
"""
import os

import numpy as np
import pandas as pd

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
RNG = np.random.default_rng(0)
NAMES = {"xlsr300m": "facebook/wav2vec2-xls-r-300m", "mms300m": "facebook/mms-300m",
         "w2vbert2": "facebook/w2v-bert-2.0", "whisper": "openai/whisper-large-v3",
         "wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ls960-ft"}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def participation_ratio(X):
    """Effective number of directions carrying variance: (sum l)^2 / sum l^2."""
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / np.maximum((ev ** 2).sum(), 1e-30))


def main():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    sub = sub.set_index("utt_id")

    rows = []
    for tag in sorted(NAMES):
        f = os.path.join(ROOT, f"eval/emb/{tag}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        utt = z["utt"].astype(str)
        E = np.nan_to_num(z["emb"][-1].astype(np.float32))      # final layer
        meta = sub.loc[utt]
        dur = meta.duration_s.values
        spk = meta.speaker_id.values
        grp = meta.content_group.values

        # ---- R2: does similarity track the duration gap?
        n = len(E)
        i = RNG.integers(0, n, 20000)
        j = RNG.integers(0, n, 20000)
        keep = (i != j) & (spk[i] != spk[j]) & (grp[i] != grp[j])
        i, j = i[keep], j[keep]
        A = l2(E)
        sim = (A[i] * A[j]).sum(1)
        dgap = np.abs(dur[i] - dur[j])
        r_dur = float(np.corrcoef(sim, dgap)[0, 1])

        # ---- R3: anisotropy, raw and centred
        mean_rand = float(sim.mean())
        Ec = E - E.mean(0, keepdims=True)
        Ac = l2(Ec)
        mean_rand_c = float((Ac[i] * Ac[j]).sum(1).mean())

        # same speaker / same content reference points
        ii = RNG.integers(0, n, 40000); jj = RNG.integers(0, n, 40000)
        ss = (ii != jj) & (spk[ii] == spk[jj])
        cc = (ii != jj) & (grp[ii] == grp[jj]) & (grp[ii] != "")
        mean_spk = float((A[ii[ss]] * A[jj[ss]]).sum(1).mean()) if ss.any() else np.nan
        mean_cnt = float((A[ii[cc]] * A[jj[cc]]).sum(1).mean()) if cc.any() else np.nan

        pr = participation_ratio(E[RNG.choice(n, min(3000, n), replace=False)])

        rows.append(dict(
            model=NAMES[tag], tag=tag, dim=E.shape[1],
            mean_cos_random=mean_rand, mean_cos_same_speaker=mean_spk,
            mean_cos_same_content=mean_cnt,
            mean_cos_random_centred=mean_rand_c,
            spread_raw=mean_spk - mean_rand,
            spread_centred=None,
            participation_ratio=pr,
            pr_fraction_of_dim=pr / E.shape[1],
            corr_sim_vs_duration_gap=r_dur,
        ))
        print(f"  {NAMES[tag]:34s} cos(rand)={mean_rand:.3f} "
              f"centred={mean_rand_c:+.3f}  PR={pr:.1f}/{E.shape[1]} "
              f"({pr/E.shape[1]:.1%})  r(sim,dur gap)={r_dur:+.3f}")
        del z, E

    df = pd.DataFrame(rows).drop(columns=["spread_centred"])
    out = os.path.join(ROOT, "eval/analysis/R_pooling_audit.tsv")
    df.to_csv(out, sep="\t", index=False)
    print("\n## R. Anisotropy and duration audit (final layer)\n")
    print(df[["model", "mean_cos_random", "mean_cos_same_speaker",
              "mean_cos_same_content", "mean_cos_random_centred",
              "participation_ratio", "pr_fraction_of_dim",
              "corr_sim_vs_duration_gap"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  mean_cos_random near 1 with a small participation ratio means the\n"
          "  embeddings sit in a narrow cone: absolute cosine is uninformative.\n"
          "  Centring removes the shared direction; if cos(random) drops to near\n"
          "  zero the cone was an offset, not a loss of information.\n"
          "  corr_sim_vs_duration_gap near 0 means similarity does NOT track clip\n"
          "  length, which is what a correct padding mask predicts.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
