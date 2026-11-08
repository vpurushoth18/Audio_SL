#!/usr/bin/env python3
"""Recompute every expansion analysis on a validated subset of checkpoints.

Writes Z-prefixed outputs beside the originals. The X*/Y*/expansion_* files are
NOT touched, so the pre-audit numbers remain on record for comparison.

Two subsets are reported side by side:

  VALIDATED   every checkpoint except those whose loader was proved invalid.
  DEDUPED     the validated set with exact-duplicate representation vectors
              collapsed to one representative. Checkpoints whose score vector
              is byte-identical across all tasks carry the same embeddings, so
              counting them separately inflates the apparent sample size and
              double-weights whatever objective label they happen to have.

The deduped set is the honest denominator for anything that averages over
checkpoints - objective-group means above all.

Usage:
  python eval/revalidated_analysis.py --exclude-file <file with one model/line>

Outputs (eval/analysis/):
  Z0_validation_inventory.tsv      per-checkpoint status, duplicate group
  Z1_transfer.tsv                  transfer, both subsets, family-blocked CIs
  Z2_group_separation.tsv          the headline, both subsets
  Z3_leave_one_family_out.tsv      LOFO on the deduped set
  Z4_capability_rankings.tsv       rankings on the validated set
  Z5_objective_summary.tsv         objective means, both subsets
  Z6_maeb_selection.tsv            MAEB-guided selection, deduped
  Z7_comparison_vs_preaudit.tsv    every headline number, before vs after
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expansion_analysis as EA

A = EA.A
N_BOOT = 2000
N_SUB = 500
SEED = 0
KEY_A = "same corpus, different capability"
KEY_B = "different corpus, same capability"


def group_of(a, b):
    sc = EA.CORPUS[a] == EA.CORPUS[b]
    scap = EA.CAPABILITY[a] == EA.CAPABILITY[b]
    return ("same corpus, same capability" if sc and scap
            else KEY_A if sc
            else KEY_B if scap
            else "different corpus, different capability")


def sp(x, y):
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return np.nan
    return stats.spearmanr(x, y).statistic


def pct(v):
    v = np.asarray([s for s in v if np.isfinite(s)])
    if v.size < 50:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fam_draws(fam, rng, n):
    fams = np.unique(fam)
    by = {f: np.flatnonzero(fam == f) for f in fams}
    return [np.concatenate([by[f] for f in rng.choice(fams, len(fams), True)])
            for _ in range(n)]


def oneper_draws(fam, rng, n):
    fams = np.unique(fam)
    by = {f: np.flatnonzero(fam == f) for f in fams}
    return [np.array([rng.choice(by[f]) for f in fams]) for _ in range(n)]


def analyse(W, C, fam, label, rng):
    """Transfer + task pairs + separation for one subset."""
    out = {}
    fb = fam_draws(fam, rng, N_BOOT)
    op = oneper_draws(fam, rng, N_SUB)

    # ---- transfer
    shared = [t for t in C.columns if C[t].notna().all()]
    cap = {c: W[[t for t in W.columns if EA.CAPABILITY.get(t) == c]].mean(axis=1)
           for c in ("speaker", "content")}
    rows = []
    for mcap, ts in [("all", shared)] + [
            (c, [t for t in shared if EA.MAEB_CAP.get(t) == c])
            for c in ("speaker", "content", "non-speech")]:
        if len(ts) < 3:
            continue
        x = C[ts].mean(axis=1).values
        for scap in ("speaker", "content"):
            y = cap[scap].values
            point = sp(x, y)
            lo, hi = pct([sp(x[i], y[i]) for i in fb])
            olo, ohi = pct([sp(x[i], y[i]) for i in op])
            rows.append(dict(subset=label, maeb_subset=mcap, n_maeb_tasks=len(ts),
                             sinaeb_capability=scap, n_checkpoints=len(W),
                             n_families=int(len(np.unique(fam))),
                             spearman=point, fam_ci_lo=lo, fam_ci_hi=hi,
                             fam_excludes_zero=bool(np.isfinite(lo) and
                                                    (lo > 0 or hi < 0)),
                             oneper_pctile_lo=olo, oneper_pctile_hi=ohi))
    out["transfer"] = pd.DataFrame(rows)

    # ---- task pairs / separation
    tasks = list(W.columns)
    V = {t: W[t].values for t in tasks}
    pairs = [(a, b) for i, a in enumerate(tasks) for b in tasks[i + 1:]
             if not (np.isnan(V[a]).any() or np.isnan(V[b]).any())]
    groups = np.array([group_of(a, b) for a, b in pairs])
    ia, ib = groups == KEY_A, groups == KEY_B

    def allp(idx):
        return np.array([sp(V[a][idx], V[b][idx]) for a, b in pairs])

    point = np.array([sp(V[a], V[b]) for a, b in pairs])
    fbm = np.array([allp(i) for i in fb])
    opm = np.array([allp(i) for i in op])

    def sep(v):
        return np.nanmedian(v[ia]) - np.nanmedian(v[ib])

    s_pt = sep(point)
    s_fb = np.array([sep(fbm[k]) for k in range(len(fbm))])
    s_op = np.array([sep(opm[k]) for k in range(len(opm))])
    flo, fhi = pct(s_fb)
    olo, ohi = pct(s_op)
    out["separation"] = pd.DataFrame([dict(
        subset=label, n_checkpoints=len(W),
        n_families=int(len(np.unique(fam))),
        median_A=np.nanmedian(point[ia]), n_pairs_A=int(ia.sum()),
        median_B=np.nanmedian(point[ib]), n_pairs_B=int(ib.sum()),
        separation=s_pt, fam_ci_lo=flo, fam_ci_hi=fhi,
        frac_resamples_negative=float(np.mean(s_fb < 0)),
        oneper_median=float(np.nanmedian(s_op)),
        oneper_pctile_lo=olo, oneper_pctile_hi=ohi,
        survives=bool(np.isfinite(fhi) and fhi < 0))])

    # ---- LOFO
    lr = []
    for f in np.unique(fam):
        keep = np.flatnonzero(fam != f)
        if len(np.unique(fam[keep])) < 2:
            continue
        v = allp(keep)
        lr.append(dict(subset=label, dropped_family=f,
                       n_dropped=int((fam == f).sum()), n_remaining=len(keep),
                       separation=sep(v), delta_vs_full=sep(v) - s_pt))
    out["lofo"] = pd.DataFrame(lr).sort_values("separation")
    out["cap"] = cap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-file", required=True)
    ap.add_argument("--corrected-sin-cache",
                    default="simteb_audio/mteb_cache_corrected")
    ap.add_argument("--corrected-maeb-cache",
                    default="simteb_audio/maeb_cache_corrected")
    a = ap.parse_args()

    excl = [l.strip() for l in open(a.exclude_file) if l.strip()
            and not l.startswith("#")]
    print(f"  excluding {len(excl)} checkpoint(s) as invalid:")
    for m in excl:
        print(f"    - {m}")

    W_all = pd.read_csv(os.path.join(A, "expansion_scores.tsv"), sep="\t",
                        index_col=0)
    maeb = EA.scrape(os.path.join(EA.ROOT, "simteb_audio/maeb_cache"))

    # ---------- overlay corrected results, where they exist and are complete
    # The original caches are left untouched; the corrected rerun wrote to its
    # own directories. A model is overlaid only if its corrected run produced
    # the full task set, so a partial rerun cannot silently half-replace it.
    overlaid = []
    cs = os.path.join(EA.ROOT, a.corrected_sin_cache)
    cm = os.path.join(EA.ROOT, a.corrected_maeb_cache)
    if os.path.isdir(cs) and os.path.isdir(cm):
        csin = EA.scrape(cs)
        csin = csin[csin.task.isin(EA.CAPABILITY)]
        cmaeb = EA.scrape(cm)
        need_sin = set(W_all.columns)
        need_maeb = set(maeb.task.unique())
        for m in sorted(set(csin.model) & set(cmaeb.model)):
            got_s = set(csin[csin.model == m].task)
            got_m = set(cmaeb[cmaeb.model == m].task)
            if need_sin <= got_s and need_maeb <= got_m:
                row = (csin[csin.model == m].set_index("task").score
                       .reindex(W_all.columns))
                W_all.loc[m] = row.values
                maeb = pd.concat([maeb[maeb.model != m],
                                  cmaeb[cmaeb.model == m]], ignore_index=True)
                overlaid.append(m)
            else:
                print(f"  corrected run INCOMPLETE for {m}: "
                      f"SinAEB {len(got_s)}/{len(need_sin)} "
                      f"MAEB {len(got_m)}/{len(need_maeb)} - not overlaid")
    print(f"  overlaid corrected results for {len(overlaid)} checkpoint(s): "
          f"{', '.join(overlaid) if overlaid else 'none'}")
    globals()["_OVERLAID"] = overlaid
    inv = pd.read_csv(os.path.join(A, "expansion_inventory.tsv"), sep="\t")
    meta = inv.set_index("model")

    # ---------- validated subset
    W = W_all.drop(index=[m for m in excl if m in W_all.index])
    C = (maeb[maeb.model.isin(W.index)]
         .pivot_table(index="model", columns="task", values="score") * 100
         ).reindex(W.index)

    # ---------- duplicate detection on the SinAEB+MAEB vector jointly
    J = pd.concat([W, C], axis=1)
    sig = J.round(12).astype(str).agg("|".join, axis=1)
    grp = {}
    for m, s in sig.items():
        grp.setdefault(s, []).append(m)
    dup_groups = [g for g in grp.values() if len(g) > 1]
    # representative = first by inventory wave then name, deterministic
    def rep_key(m):
        return (int(meta.loc[m, "wave"]) if m in meta.index else 99, m)
    reps = {}
    for g in grp.values():
        r = sorted(g, key=rep_key)[0]
        for m in g:
            reps[m] = r

    inv_rows = []
    for m in W_all.index:
        g = [x for x in grp.get(sig.get(m, "NA"), []) ] if m in sig.index else []
        inv_rows.append(dict(
            model=m,
            corrected_rerun_overlaid=m in _OVERLAID,
            family=meta.loc[m, "family"] if m in meta.index else None,
            objective=meta.loc[m, "objective"] if m in meta.index else None,
            status="EXCLUDED_INVALID_LOADER" if m in excl else "VALIDATED",
            duplicate_group_size=len(g),
            duplicate_group=("; ".join(sorted(g)) if len(g) > 1 else ""),
            is_duplicate_representative=(reps.get(m) == m) if m in reps else None,
            kept_in_deduped=(m not in excl and reps.get(m) == m)))
    z0 = pd.DataFrame(inv_rows)
    z0.to_csv(os.path.join(A, "Z0_validation_inventory.tsv"), sep="\t",
              index=False)
    print(f"\n  validated: {len(W)}   duplicate groups: {len(dup_groups)}")
    for g in dup_groups:
        print(f"    identical scores: {', '.join(sorted(g))}")

    keep_dedup = [m for m in W.index if reps[m] == m]
    Wd, Cd = W.loc[keep_dedup], C.loc[keep_dedup]
    print(f"  deduped:   {len(Wd)}")

    famv = np.array([meta.loc[m, "family"] for m in W.index])
    famd = np.array([meta.loc[m, "family"] for m in Wd.index])

    rng = np.random.default_rng(SEED)
    res_v = analyse(W, C, famv, "validated", rng)
    rng = np.random.default_rng(SEED)
    res_d = analyse(Wd, Cd, famd, "deduped", rng)

    pd.concat([res_v["transfer"], res_d["transfer"]]).to_csv(
        os.path.join(A, "Z1_transfer.tsv"), sep="\t", index=False,
        float_format="%.17g")
    sepdf = pd.concat([res_v["separation"], res_d["separation"]])
    sepdf.to_csv(os.path.join(A, "Z2_group_separation.tsv"), sep="\t",
                 index=False, float_format="%.17g")
    pd.concat([res_v["lofo"], res_d["lofo"]]).to_csv(
        os.path.join(A, "Z3_leave_one_family_out.tsv"), sep="\t", index=False,
        float_format="%.17g")

    print("\n  === SEPARATION ===")
    print(sepdf[["subset", "n_checkpoints", "n_families", "median_A", "median_B",
                 "separation", "fam_ci_lo", "fam_ci_hi",
                 "frac_resamples_negative", "survives"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  === TRANSFER (deduped) ===")
    print(res_d["transfer"][["maeb_subset", "sinaeb_capability", "spearman",
                             "fam_ci_lo", "fam_ci_hi", "fam_excludes_zero"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------- rankings on validated; objective summary on both
    def rank_table(Wx, Cx, label):
        cap = {c: Wx[[t for t in Wx.columns if EA.CAPABILITY.get(t) == c]]
               .mean(axis=1) for c in ("speaker", "content")}
        d = pd.DataFrame({"speaker": cap["speaker"], "content": cap["content"]})
        for col in ("family", "objective", "encoder_params_m"):
            d[col] = [meta.loc[m, col] for m in d.index]
        d["subset"] = label
        d["rank_speaker"] = d.speaker.rank(ascending=False, method="min").astype(int)
        d["rank_content"] = d.content.rank(ascending=False, method="min").astype(int)
        return d, cap

    dv, capv = rank_table(W, C, "validated")
    dd, capd = rank_table(Wd, Cd, "deduped")
    pd.concat([dv, dd]).to_csv(os.path.join(A, "Z4_capability_rankings.tsv"),
                               sep="\t", float_format="%.17g")
    objs = []
    for d, lab in ((dv, "validated"), (dd, "deduped")):
        g = (d.groupby("objective")
               .agg(n=("speaker", "size"), speaker_mean=("speaker", "mean"),
                    content_mean=("content", "mean")))
        g["subset"] = lab
        objs.append(g.reset_index())
    z5 = pd.concat(objs)
    z5.to_csv(os.path.join(A, "Z5_objective_summary.tsv"), sep="\t", index=False,
              float_format="%.17g")
    print("\n  === OBJECTIVE GROUPS ===")
    for lab in ("validated", "deduped"):
        print(f"\n  {lab}:")
        print(z5[z5.subset == lab][["objective", "n", "speaker_mean",
                                    "content_mean"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------- MAEB-guided selection on deduped
    shared = [t for t in Cd.columns if Cd[t].notna().all()]
    rk = {c: capd[c].rank(ascending=False, method="min")
          for c in ("speaker", "content")}
    rows = []
    for name, ts in [("all", shared)] + [
            (c, [t for t in shared if EA.MAEB_CAP.get(t) == c])
            for c in ("speaker", "content", "non-speech")]:
        if len(ts) < 3:
            continue
        x = Cd[ts].mean(axis=1)
        for k in (1, 3, 5):
            pick = x.nlargest(k).index.tolist()
            rows.append(dict(subset="deduped", maeb_subset=name, top_k=k,
                             picked="; ".join(pick),
                             mean_speaker_rank=float(rk["speaker"][pick].mean()),
                             mean_content_rank=float(rk["content"][pick].mean()),
                             random_baseline=(len(Wd) + 1) / 2,
                             n_checkpoints=len(Wd)))
    z6 = pd.DataFrame(rows)
    z6.to_csv(os.path.join(A, "Z6_maeb_selection.tsv"), sep="\t", index=False,
              float_format="%.17g")
    print("\n  === MAEB-GUIDED SELECTION (deduped, top-3) ===")
    print(z6[z6.top_k == 3][["maeb_subset", "mean_speaker_rank",
                             "mean_content_rank", "random_baseline"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # ---------- before/after comparison
    old = pd.read_csv(os.path.join(A, "X3_group_separation.tsv"), sep="\t")
    oldsep = float(old.loc[old.statistic.str.startswith("separation"),
                           "point"].iloc[0])
    oldtr = pd.read_csv(os.path.join(A, "X1_transfer_family_blocked.tsv"), sep="\t")
    cmp_rows = [dict(quantity="group separation", preaudit_45=oldsep,
                     validated=float(res_v["separation"].separation.iloc[0]),
                     deduped=float(res_d["separation"].separation.iloc[0]))]
    for _, r in oldtr.iterrows():
        def pick(df):
            m = df[(df.maeb_subset == r.maeb_subset) &
                   (df.sinaeb_capability == r.sinaeb_capability)]
            return float(m.spearman.iloc[0]) if len(m) else np.nan
        cmp_rows.append(dict(
            quantity=f"transfer {r.maeb_subset}->{r.sinaeb_capability}",
            preaudit_45=r.spearman, validated=pick(res_v["transfer"]),
            deduped=pick(res_d["transfer"])))
    rc_old = pd.read_csv(os.path.join(A, "Y4_rank_correlation.tsv"), sep="\t")
    cmp_rows.append(dict(
        quantity="speaker-vs-content rank rho",
        preaudit_45=float(rc_old.spearman.iloc[0]),
        validated=float(stats.spearmanr(dv.speaker, dv.content).statistic),
        deduped=float(stats.spearmanr(dd.speaker, dd.content).statistic)))
    z7 = pd.DataFrame(cmp_rows)
    z7["delta_dedup_vs_preaudit"] = z7.deduped - z7.preaudit_45
    z7.to_csv(os.path.join(A, "Z7_comparison_vs_preaudit.tsv"), sep="\t",
              index=False, float_format="%.17g")
    print("\n  === BEFORE / AFTER ===")
    print(z7.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    json.dump(dict(excluded=excl, overlaid_corrected=_OVERLAID,
                   n_validated=len(W), n_deduped=len(Wd),
                   n_families_validated=int(len(set(famv))),
                   n_families_deduped=int(len(set(famd))),
                   duplicate_groups=[sorted(g) for g in dup_groups],
                   n_boot=N_BOOT, n_oneper=N_SUB, seed=SEED),
              open(os.path.join(A, "Z_revalidation_config.json"), "w"), indent=2)
    print("\n  wrote Z0-Z7 and Z_revalidation_config.json")


if __name__ == "__main__":
    main()
