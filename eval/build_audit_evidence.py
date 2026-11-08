#!/usr/bin/env python3
"""Stage the loading-audit evidence package.

Read-only with respect to results. The original caches, the pre-audit analysis
files (expansion_*, X*, Y*) and the corrected caches are all copied, never
modified.

Layout:
  01_diagnostics/            the audit itself: findings, raw JSON, logs
  02_code_changes/           the corrected loader, the registry hook, the diff
  03_corrected_raw_results/  every JSON from the two corrected reruns
  04_corrected_scores/       consolidated unrounded scores, before vs after
  05_revalidated_analysis/   Z0-Z7 on the validated and deduplicated sets
  06_preaudit_for_comparison/ the superseded numbers, kept for diffing
  07_reproducibility/        inventory, environment, commit, patch
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expansion_analysis as EA

ROOT = EA.ROOT
A = EA.A
AFFECTED = ["facebook/wav2vec2-conformer-rel-pos-large",
            "facebook/wav2vec2-xls-r-300m-21-to-en"]
DUPS = [["microsoft/wavlm-base", "microsoft/wavlm-base-sv"],
        ["microsoft/wavlm-base-plus", "microsoft/wavlm-base-plus-sd",
         "microsoft/wavlm-base-plus-sv"]]


def main():
    stage = sys.argv[1]
    audit_dir = sys.argv[2]
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    missing = []

    def put(rel):
        p = os.path.join(stage, rel)
        os.makedirs(p, exist_ok=True)
        return p

    def cp(src, dst, name=None):
        s = src if os.path.isabs(src) else os.path.join(ROOT, src)
        if not os.path.exists(s):
            missing.append(src)
            return False
        shutil.copy2(s, os.path.join(dst, name or os.path.basename(s)))
        return True

    # ---------------- 01 diagnostics
    d1 = put("01_diagnostics")
    cp(os.path.join(audit_dir, "loading_audit.json"), d1)
    cp("eval/loading_audit.py", d1)
    # the authoritative missing-key evidence lives in the original worker log
    ev = put("01_diagnostics/original_run_log_excerpts")
    for m in AFFECTED:
        tag = m.split("/")[1]
        out = os.path.join(ev, f"{tag}__missing_keys_from_expansion_gpu1.log")
        try:
            txt = open(os.path.join(ROOT, "logs/expansion_gpu1.log"),
                       errors="replace").read()
            i = txt.find(m)
            seg = txt[i:i + 20000] if i >= 0 else "MARKER NOT FOUND"
            open(out, "w").write(seg)
        except Exception as e:                          # noqa: BLE001
            missing.append(f"log excerpt for {m}: {e}")

    # ---------------- 02 code changes
    d2 = put("02_code_changes")
    for f in ["simteb_audio/src/simteb_audio/models/corrected_loaders.py",
              "simteb_audio/src/simteb_audio/models/__init__.py",
              "jobs/rerun_corrected.sh", "jobs/audit_followup_chain.sh",
              "eval/revalidated_analysis.py", "eval/build_audit_evidence.py"]:
        cp(f, d2)
    try:
        diff = subprocess.run(
            ["git", "diff", "--",
             "simteb_audio/src/simteb_audio/models/__init__.py"],
            cwd=ROOT, capture_output=True, text=True, timeout=120).stdout
        open(os.path.join(d2, "models_init.diff"), "w").write(diff)
    except Exception:
        missing.append("git diff for models/__init__.py")

    # ---------------- 03 corrected raw results
    d3 = put("03_corrected_raw_results")
    n_raw = 0
    for bench, cache in (("sinaeb", "simteb_audio/mteb_cache_corrected"),
                         ("maeb", "simteb_audio/maeb_cache_corrected")):
        src = os.path.join(ROOT, cache, "results")
        if not os.path.isdir(src):
            missing.append(cache)
            continue
        shutil.copytree(src, os.path.join(d3, bench))
        n_raw += len([f for f in glob.glob(os.path.join(d3, bench, "**", "*.json"),
                                           recursive=True)
                      if os.path.basename(f) != "model_meta.json"])

    # ---------------- 04 corrected scores, before vs after
    d4 = put("04_corrected_scores")
    csin = EA.scrape(os.path.join(ROOT, "simteb_audio/mteb_cache_corrected"))
    cmaeb = EA.scrape(os.path.join(ROOT, "simteb_audio/maeb_cache_corrected"))
    csin = csin[csin.task.isin(EA.CAPABILITY)]
    csin["benchmark"], cmaeb["benchmark"] = "SinAEB", "MAEB"
    pd.concat([csin, cmaeb], ignore_index=True).to_csv(
        os.path.join(d4, "corrected_scores_long.tsv"), sep="\t", index=False,
        float_format="%.17g")

    old_sin = EA.scrape(os.path.join(ROOT, "simteb_audio/mteb_cache"))
    old_sin = old_sin[old_sin.task.isin(EA.CAPABILITY)]
    rows = []
    for m in AFFECTED:
        o = old_sin[old_sin.model == m].set_index("task").score
        n = csin[csin.model == m].set_index("task").score
        for t in sorted(set(o.index) | set(n.index)):
            rows.append(dict(model=m, benchmark="SinAEB", task=t,
                             score_before=o.get(t), score_after=n.get(t),
                             delta=(n.get(t) - o.get(t))
                             if (t in o.index and t in n.index) else None))
        om = EA.scrape(os.path.join(ROOT, "simteb_audio/maeb_cache"))
        om = om[om.model == m].set_index("task").score
        nm = cmaeb[cmaeb.model == m].set_index("task").score
        for t in sorted(set(om.index) | set(nm.index)):
            rows.append(dict(model=m, benchmark="MAEB", task=t,
                             score_before=om.get(t), score_after=nm.get(t),
                             delta=(nm.get(t) - om.get(t))
                             if (t in om.index and t in nm.index) else None))
    ba = pd.DataFrame(rows)
    ba.to_csv(os.path.join(d4, "before_after_per_task.tsv"), sep="\t",
              index=False, float_format="%.17g")

    # ---------------- 05 revalidated analysis
    d5 = put("05_revalidated_analysis")
    for f in ["Z0_validation_inventory.tsv", "Z1_transfer.tsv",
              "Z2_group_separation.tsv", "Z3_leave_one_family_out.tsv",
              "Z4_capability_rankings.tsv", "Z5_objective_summary.tsv",
              "Z6_maeb_selection.tsv", "Z7_comparison_vs_preaudit.tsv",
              "Z_revalidation_config.json"]:
        cp(f"eval/analysis/{f}", d5)
    cp("logs/revalidated_analysis.log", d5)

    # ---------------- 06 pre-audit, for comparison
    d6 = put("06_preaudit_for_comparison")
    for f in ["expansion_scores.tsv", "expansion_status.tsv",
              "expansion_transfer.tsv", "expansion_task_corr.tsv",
              "expansion_group_summary.tsv",
              "X1_transfer_family_blocked.tsv", "X3_group_separation.tsv",
              "X4_leave_one_family_out.tsv", "Y1_capability_rankings.tsv",
              "Y2_objective_group_summary.tsv", "Y3_maeb_guided_selection.tsv",
              "Y4_rank_correlation.tsv"]:
        cp(f"eval/analysis/{f}", d6)

    # ---------------- 07 reproducibility
    d7 = put("07_reproducibility")
    cp("eval/analysis/expansion_inventory.tsv", d7)
    for f in ["logs/rerun_conformer.log", "logs/rerun_21toen.log",
              "logs/audit_followup_chain.log"]:
        cp(f, d7)
    env = {}
    for m in ["torch", "transformers", "mteb", "numpy", "scipy", "pandas",
              "sklearn"]:
        try:
            env[m] = getattr(__import__(m), "__version__", "unknown")
        except Exception:
            env[m] = "NOT INSTALLED"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True,
                                timeout=60).stdout.strip()
    except Exception:
        commit = "unavailable"
    json.dump(dict(packages=env, python=sys.version.split()[0],
                   git_commit=commit,
                   corrected_rerun_window="2026-10-07 22:38 to 2026-10-08 03:06 PDT",
                   loader_flag="SIMTEB_CORRECTED_LOADERS=1",
                   corrected_caches=["simteb_audio/mteb_cache_corrected",
                                     "simteb_audio/maeb_cache_corrected"],
                   affected_checkpoints=AFFECTED,
                   duplicate_groups=DUPS),
              open(os.path.join(d7, "audit_environment.json"), "w"), indent=2)

    # ---------------- manifest
    rows = []
    for dp, _, fs in os.walk(stage):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            rows.append(dict(path=os.path.relpath(p, stage),
                             bytes=os.path.getsize(p),
                             sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()))
    man = pd.DataFrame(rows).sort_values("path")
    man.to_csv(os.path.join(stage, "MANIFEST_sha256.tsv"), sep="\t", index=False)
    print(f"  staged {len(man)} files, {man.bytes.sum()/1e6:.1f} MB")
    print(f"  corrected raw JSONs: {n_raw}")
    if missing:
        print(f"  NOT FOUND ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")
    json.dump(dict(staged=len(man), bytes=int(man.bytes.sum()),
                   corrected_raw_jsons=n_raw, missing=missing),
              open(os.path.join(stage, "_build_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
