#!/usr/bin/env python3
"""Family-blocked (cluster) bootstrap for the expansion correlations.

The checkpoint bootstrap in expansion_analysis.py resamples checkpoints
independently. They are not independent: of the usable checkpoints roughly a
quarter are Whisper sizes, and several families contribute five or six
checkpoints each. Sizes of one model share an architecture, a pretraining
corpus and a tokenisation of the audio, so they carry much less than one
checkpoint's worth of independent information each. Resampling them
independently therefore understates every interval.

This recomputes the same statistics under three stricter treatments:

  1. FAMILY-BLOCKED BOOTSTRAP. Resample FAMILIES with replacement, taking every
     checkpoint of each family drawn. The resampling unit becomes the family,
     so within-family dependence no longer inflates precision. This is the
     headline correction; the number of clusters (not checkpoints) sets the
     effective sample size.

  2. LEAVE-ONE-FAMILY-OUT. Drop each family in turn and recompute. Shows
     whether any single family drives a result -- in particular whether the ten
     Whisper checkpoints are doing the work.

  3. ONE-PER-FAMILY SUBSAMPLE. Draw exactly one checkpoint per family, giving a
     set with no within-family dependence at all, and repeat over random
     draws. The strictest test, at the cost of a much smaller n.

Statistics covered:
  - the 8 MAEB -> SinAEB transfer cells
  - every within-SinAEB task pair
  - the group medians and, the paper's headline, the separation between
    "same corpus, different capability" and "different corpus, same capability"

Nothing here re-runs a model; it reads expansion_scores.tsv and the MAEB cache.

Outputs:
  X1_transfer_family_blocked.tsv     transfer, 3 treatments side by side
  X2_task_pairs_family_blocked.tsv   every pair, checkpoint vs family CI
  X3_group_separation.tsv            the headline separation under each treatment
  X4_leave_one_family_out.tsv        per-family influence on the separation
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expansion_analysis as EA          # reuse scrape/CAPABILITY/MAEB_CAP/CORPUS

ROOT = EA.ROOT
A = EA.A
N_BOOT = 2000
N_SUB = 500            # one-per-family draws
SEED = 0


# --------------------------------------------------------------------------
def spearman(x, y):
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return np.nan
    return stats.spearmanr(x, y).statistic


def family_blocked_indices(fam_of_row, rng):
    """Resample families with replacement; return row indices of the draw."""
    fams = np.unique(fam_of_row)
    by_fam = {f: np.flatnonzero(fam_of_row == f) for f in fams}
    drawn = rng.choice(fams, size=len(fams), replace=True)
    return np.concatenate([by_fam[f] for f in drawn])


def one_per_family_indices(fam_of_row, rng):
    fams = np.unique(fam_of_row)
    return np.array([rng.choice(np.flatnonzero(fam_of_row == f)) for f in fams])


def pct(v):
    v = np.asarray([s for s in v if np.isfinite(s)])
    if v.size < 50:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    W = pd.read_csv(os.path.join(A, "expansion_scores.tsv"), sep="\t",
                    index_col=0)
    maeb = EA.scrape(os.path.join(ROOT, "simteb_audio/maeb_cache"))
    C = (maeb[maeb.model.isin(W.index)]
         .pivot_table(index="model", columns="task", values="score") * 100)
    C = C.reindex(W.index)

    inv = pd.read_csv(os.path.join(A, "expansion_inventory.tsv"), sep="\t")
    fam_map = dict(zip(inv.model, inv.family))
    missing = [m for m in W.index if m not in fam_map]
    if missing:
        print(f"  WARNING: no family for {missing}; excluded")
        W = W.drop(index=missing)
        C = C.drop(index=missing)
    fam = np.array([fam_map[m] for m in W.index])

    M = len(W)
    fams, counts = np.unique(fam, return_counts=True)
    print(f"  checkpoints: {M}   families: {len(fams)}")
    print("  family sizes: " + ", ".join(f"{f}={c}" for f, c in zip(fams, counts)))
    print(f"  largest family is {counts.max()}/{M} = {counts.max()/M:.0%} "
          f"of the checkpoints\n")

    # pre-draw the index sets once so every statistic uses the SAME resamples
    fb_idx = [family_blocked_indices(fam, rng) for _ in range(N_BOOT)]
    sub_idx = [one_per_family_indices(fam, rng) for _ in range(N_SUB)]

    # ================= 1. transfer =================
    shared = [t for t in C.columns if C[t].notna().all()]
    cap_mean = {c: W[[t for t in W.columns if EA.CAPABILITY.get(t) == c]].mean(axis=1)
                for c in ("speaker", "content")}
    subsets = [("all", shared)] + [
        (c, [t for t in shared if EA.MAEB_CAP.get(t) == c])
        for c in ("speaker", "content", "non-speech")]

    rows = []
    for mcap, tsk in subsets:
        if len(tsk) < 3:
            continue
        x = C[tsk].mean(axis=1).values
        for scap in ("speaker", "content"):
            y = cap_mean[scap].values
            point = spearman(x, y)
            fb = [spearman(x[i], y[i]) for i in fb_idx]
            sb = [spearman(x[i], y[i]) for i in sub_idx]
            flo, fhi = pct(fb)
            slo, shi = pct(sb)
            rows.append(dict(
                maeb_subset=mcap, n_maeb_tasks=len(tsk), sinaeb_capability=scap,
                n_checkpoints=M, n_families=len(fams), spearman=point,
                fam_ci_lo=flo, fam_ci_hi=fhi,
                fam_excludes_zero=bool(np.isfinite(flo) and flo > 0),
                oneper_median=float(np.nanmedian(sb)),
                oneper_ci_lo=slo, oneper_ci_hi=shi))
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join(A, "X1_transfer_family_blocked.tsv"), sep="\t",
              index=False, float_format="%.17g")

    old = pd.read_csv(os.path.join(A, "expansion_transfer.tsv"), sep="\t")
    cmp_ = tr.merge(old[["maeb_subset", "sinaeb_capability", "ci_lo", "ci_hi",
                         "excludes_zero"]],
                    on=["maeb_subset", "sinaeb_capability"])
    cmp_["ckpt_width"] = cmp_.ci_hi - cmp_.ci_lo
    cmp_["fam_width"] = cmp_.fam_ci_hi - cmp_.fam_ci_lo
    cmp_["widening"] = cmp_.fam_width / cmp_.ckpt_width
    print("  === 1. TRANSFER: checkpoint vs family-blocked ===")
    print(cmp_[["maeb_subset", "sinaeb_capability", "spearman", "ci_lo", "ci_hi",
                "fam_ci_lo", "fam_ci_hi", "widening", "excludes_zero",
                "fam_excludes_zero"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  cells excluding zero: checkpoint {int(cmp_.excludes_zero.sum())}/8"
          f"  ->  family-blocked {int(cmp_.fam_excludes_zero.sum())}/8")
    print(f"  mean CI widening under family blocking: "
          f"{cmp_.widening.mean():.2f}x\n")

    # ================= 2. task pairs + 3. groups =================
    tasks = list(W.columns)
    Wv = {t: W[t].values for t in tasks}
    pairs = [(a, b) for i, a in enumerate(tasks) for b in tasks[i + 1:]
             if not (np.isnan(Wv[a]).any() or np.isnan(Wv[b]).any())]

    def group_of(a, b):
        sc = EA.CORPUS[a] == EA.CORPUS[b]
        scap = EA.CAPABILITY[a] == EA.CAPABILITY[b]
        return ("same corpus, same capability" if sc and scap
                else "same corpus, different capability" if sc
                else "different corpus, same capability" if scap
                else "different corpus, different capability")

    groups = np.array([group_of(a, b) for a, b in pairs])
    KEY_A = "same corpus, different capability"
    KEY_B = "different corpus, same capability"
    ia, ib = groups == KEY_A, groups == KEY_B

    point = np.array([spearman(Wv[a], Wv[b]) for a, b in pairs])

    def all_pairs(idx):
        return np.array([spearman(Wv[a][idx], Wv[b][idx]) for a, b in pairs])

    print(f"  === 2. TASK PAIRS: {len(pairs)} pairs, "
          f"{N_BOOT} family-blocked resamples ===")
    fb_mat = np.empty((N_BOOT, len(pairs)))
    for k, idx in enumerate(fb_idx):
        fb_mat[k] = all_pairs(idx)
        if (k + 1) % 500 == 0:
            print(f"    {k+1}/{N_BOOT}", flush=True)
    sb_mat = np.array([all_pairs(idx) for idx in sub_idx])

    tc_old = pd.read_csv(os.path.join(A, "expansion_task_corr.tsv"), sep="\t")
    key_old = {(r.task_a, r.task_b): (r.ci_lo, r.ci_hi)
               for r in tc_old.itertuples()}
    prows = []
    for j, (a, b) in enumerate(pairs):
        flo, fhi = pct(fb_mat[:, j])
        clo, chi = key_old.get((a, b), (np.nan, np.nan))
        prows.append(dict(
            task_a=a, task_b=b, group=groups[j], spearman=point[j],
            ckpt_ci_lo=clo, ckpt_ci_hi=chi, fam_ci_lo=flo, fam_ci_hi=fhi,
            ckpt_excludes_zero=bool(np.isfinite(clo) and (clo > 0 or chi < 0)),
            fam_excludes_zero=bool(np.isfinite(flo) and (flo > 0 or fhi < 0)),
            oneper_median=float(np.nanmedian(sb_mat[:, j]))))
    tp = pd.DataFrame(prows)
    tp.to_csv(os.path.join(A, "X2_task_pairs_family_blocked.tsv"), sep="\t",
              index=False, float_format="%.17g")
    print(f"\n  pairs with an interval excluding zero: "
          f"checkpoint {int(tp.ckpt_excludes_zero.sum())}/{len(tp)}"
          f"  ->  family {int(tp.fam_excludes_zero.sum())}/{len(tp)}")

    # group medians and the separation
    def sep(vec):
        return np.nanmedian(vec[ia]) - np.nanmedian(vec[ib])

    sep_point = sep(point)
    sep_fb = np.array([sep(fb_mat[k]) for k in range(N_BOOT)])
    sep_sb = np.array([sep(sb_mat[k]) for k in range(len(sub_idx))])
    flo, fhi = pct(sep_fb)
    slo, shi = pct(sep_sb)

    grows = [dict(statistic="median(same corpus, diff capability)",
                  point=np.nanmedian(point[ia]),
                  fam_ci_lo=pct(np.nanmedian(fb_mat[:, ia], axis=1))[0],
                  fam_ci_hi=pct(np.nanmedian(fb_mat[:, ia], axis=1))[1]),
             dict(statistic="median(diff corpus, same capability)",
                  point=np.nanmedian(point[ib]),
                  fam_ci_lo=pct(np.nanmedian(fb_mat[:, ib], axis=1))[0],
                  fam_ci_hi=pct(np.nanmedian(fb_mat[:, ib], axis=1))[1]),
             dict(statistic="separation (A - B)", point=sep_point,
                  fam_ci_lo=flo, fam_ci_hi=fhi)]
    gs = pd.DataFrame(grows)
    gs["n_pairs_A"], gs["n_pairs_B"] = int(ia.sum()), int(ib.sum())
    gs["n_families"] = len(fams)
    gs["oneper_median_separation"] = float(np.nanmedian(sep_sb))
    gs["oneper_ci_lo"], gs["oneper_ci_hi"] = slo, shi
    gs.to_csv(os.path.join(A, "X3_group_separation.tsv"), sep="\t",
              index=False, float_format="%.17g")

    print("\n  === 3. HEADLINE SEPARATION ===")
    print(gs[["statistic", "point", "fam_ci_lo", "fam_ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    frac = float(np.mean(sep_fb < 0))
    print(f"\n  separation point estimate      : {sep_point:+.3f}")
    print(f"  family-blocked 95% CI          : [{flo:+.3f}, {fhi:+.3f}]")
    print(f"  resamples with separation < 0  : {frac:.1%}")
    print(f"  one-per-family ({len(fams)} checkpoints) : "
          f"{np.nanmedian(sep_sb):+.3f}  [{slo:+.3f}, {shi:+.3f}]")
    print(f"  SURVIVES family blocking       : "
          f"{'YES' if np.isfinite(fhi) and fhi < 0 else 'NO'}")

    # ================= 4. leave-one-family-out =================
    lrows = []
    for f in fams:
        keep = np.flatnonzero(fam != f)
        v = all_pairs(keep)
        lrows.append(dict(dropped_family=f,
                          n_dropped=int((fam == f).sum()),
                          n_remaining=len(keep),
                          median_A=np.nanmedian(v[ia]),
                          median_B=np.nanmedian(v[ib]),
                          separation=sep(v),
                          delta_vs_full=sep(v) - sep_point))
    lo = pd.DataFrame(lrows).sort_values("separation")
    lo.to_csv(os.path.join(A, "X4_leave_one_family_out.tsv"), sep="\t",
              index=False, float_format="%.17g")
    print("\n  === 4. LEAVE-ONE-FAMILY-OUT (separation) ===")
    print(lo.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n  separation stays negative in "
          f"{int((lo.separation < 0).sum())}/{len(lo)} leave-one-family-out fits")
    print(f"  range: {lo.separation.min():+.3f} to {lo.separation.max():+.3f}")

    json.dump(dict(n_checkpoints=M, n_families=int(len(fams)),
                   family_sizes={str(f): int(c) for f, c in zip(fams, counts)},
                   n_boot=N_BOOT, n_oneper=N_SUB, seed=SEED,
                   separation_point=float(sep_point),
                   separation_family_ci=[flo, fhi],
                   survives=bool(np.isfinite(fhi) and fhi < 0)),
              open(os.path.join(A, "X_family_blocked_config.json"), "w"), indent=2)
    print("\n  wrote X1_transfer, X2_task_pairs, X3_group_separation, "
          "X4_leave_one_family_out")


if __name__ == "__main__":
    main()
