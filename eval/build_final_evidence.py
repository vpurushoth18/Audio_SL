#!/usr/bin/env python3
"""Stage the 45-checkpoint evidence package and write its metadata exports.

Read-only with respect to results: it copies out of eval/analysis and the two
result caches and writes only into the staging directory. Nothing under
simteb_audio/*_cache or eval/analysis is modified.

Produces, inside the staging tree:
  01_raw_results/              every evaluator JSON, both benchmarks
  02_consolidated_scores/      unrounded model x task matrices, long + wide
  03_analysis_final_45/        the 45-checkpoint analysis outputs
  04_robustness_family/        family-blocked / LOFO / one-per-family
  05_rankings_selection/       capability rankings, MAEB-guided selection
  06_code/                     the scripts and local packages that produced it
  07_reproducibility/          checkpoints, mappings, settings, versions, patch
  08_logs/                     run and failure logs
  09_figures_and_tables/       figures with source data and generator
  10_layer_study_6_encoders/   the completed 6-encoder layer analysis
  11_layer_extension_planned/  protocol and status ONLY - not run
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
STAGE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage_evidence"

D = dict(raw="01_raw_results", cons="02_consolidated_scores",
         ana="03_analysis_final_45", rob="04_robustness_family",
         rank="05_rankings_selection", code="06_code",
         repro="07_reproducibility", logs="08_logs",
         figs="09_figures_and_tables", layer="10_layer_study_6_encoders",
         ext="11_layer_extension_planned")


def put(rel):
    p = os.path.join(STAGE, rel)
    os.makedirs(p, exist_ok=True)
    return p


def cp(src, dstdir, name=None):
    s = os.path.join(ROOT, src) if not os.path.isabs(src) else src
    if not os.path.exists(s):
        return False
    shutil.copy2(s, os.path.join(dstdir, name or os.path.basename(s)))
    return True


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    missing = []

    # ---------------------------------------------------------------- 01 raw
    raw = put(D["raw"])
    counts = {}
    for bench, cache in (("sinaeb", "simteb_audio/mteb_cache"),
                         ("maeb", "simteb_audio/maeb_cache")):
        src = os.path.join(ROOT, cache, "results")
        dst = os.path.join(raw, bench)
        shutil.copytree(src, dst)
        n = len([f for f in glob.glob(os.path.join(dst, "**", "*.json"),
                                      recursive=True)
                 if os.path.basename(f) != "model_meta.json"])
        counts[bench] = n
    print(f"  raw JSONs copied: {counts}")

    # ---------------------------------------------------- 02 consolidated
    cons = put(D["cons"])
    sin = EA.scrape(os.path.join(ROOT, "simteb_audio/mteb_cache"))
    maeb = EA.scrape(os.path.join(ROOT, "simteb_audio/maeb_cache"))
    sin["benchmark"], maeb["benchmark"] = "SinAEB", "MAEB"
    sin["capability"] = sin.task.map(EA.CAPABILITY)
    sin["corpus"] = sin.task.map(EA.CORPUS)
    maeb["capability"] = maeb.task.map(EA.MAEB_CAP)
    long = pd.concat([sin, maeb], ignore_index=True)
    long.to_csv(os.path.join(cons, "all_scores_long.tsv"), sep="\t",
                index=False, float_format="%.17g")
    sin.pivot_table(index="model", columns="task", values="score").to_csv(
        os.path.join(cons, "sinaeb_scores_wide.tsv"), sep="\t",
        float_format="%.17g")
    maeb.pivot_table(index="model", columns="task", values="score").to_csv(
        os.path.join(cons, "maeb_scores_wide.tsv"), sep="\t",
        float_format="%.17g")
    print(f"  consolidated: {len(long)} long rows "
          f"(SinAEB {len(sin)}, MAEB {len(maeb)})")

    # ------------------------------------------------------- 03/04/05 analysis
    ana, rob, rank = put(D["ana"]), put(D["rob"]), put(D["rank"])
    for f in ["expansion_scores.tsv", "expansion_status.tsv",
              "expansion_inventory.tsv", "expansion_transfer.tsv",
              "expansion_task_corr.tsv", "expansion_group_summary.tsv",
              "expansion_state.json"]:
        if not cp(f"eval/analysis/{f}", ana):
            missing.append(f"eval/analysis/{f}")
    for f in ["X1_transfer_family_blocked.tsv", "X2_task_pairs_family_blocked.tsv",
              "X3_group_separation.tsv", "X4_leave_one_family_out.tsv",
              "X_family_blocked_config.json"]:
        if not cp(f"eval/analysis/{f}", rob):
            missing.append(f"eval/analysis/{f}")
    for f in ["Y1_capability_rankings.tsv", "Y2_objective_group_summary.tsv",
              "Y3_maeb_guided_selection.tsv", "Y4_rank_correlation.tsv"]:
        if not cp(f"eval/analysis/{f}", rank):
            missing.append(f"eval/analysis/{f}")

    # ---------------------------------------------------------------- 06 code
    code = put(D["code"])
    for f in ["eval/expansion_inventory.py", "eval/expansion_analysis.py",
              "eval/family_blocked_bootstrap.py", "eval/capability_rankings.py",
              "eval/build_final_evidence.py", "eval/figures.py",
              "simteb_audio/evaluate.py", "simteb_audio/maeb_evaluate.py"]:
        if not cp(f, code):
            missing.append(f)
    shutil.copytree(os.path.join(ROOT, "simteb_audio/src/simteb_audio"),
                    os.path.join(code, "simteb_audio_package"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    jobs = put(os.path.join(D["code"], "jobs"))
    for f in ["expansion_worker.sh", "expansion_worker5.sh", "expansion_wave4.sh",
              "expansion_wave5.sh", "family_bootstrap_chain.sh", "worker.sh",
              "worker_new.sh", "worker_maeb.sh"]:
        if not cp(f"jobs/{f}", jobs):
            missing.append(f"jobs/{f}")

    # ------------------------------------------------------ 07 reproducibility
    rp = put(D["repro"])
    inv = pd.read_csv(os.path.join(A, "expansion_inventory.tsv"), sep="\t")
    st = pd.read_csv(os.path.join(A, "expansion_status.tsv"), sep="\t")
    used = set(st[(st.complete) & (~st.is_baseline)].model)

    # checkpoint manifest with the wrapper actually used + resolved revision
    sys.path.insert(0, os.path.join(ROOT, "simteb_audio/src"))
    import simteb_audio.models.expansion_models  # noqa: F401  (registers)
    import mteb
    rows = []
    for _, r in inv.iterrows():
        rec = dict(model=r.model, family=r.family, objective=r.objective,
                   encoder_params_m=r.encoder_params_m, wave=r.wave,
                   note=r.note, used_in_analysis=r.model in used)
        try:
            mm = mteb.get_model_meta(r.model)
            ld = mm.loader
            cls = ld.func if hasattr(ld, "func") else ld
            rec.update(wrapper=cls.__name__, revision_declared=mm.revision,
                       embed_dim=mm.embed_dim, n_parameters=mm.n_parameters)
        except Exception as e:
            rec.update(wrapper=f"UNRESOLVED: {type(e).__name__}",
                       revision_declared=None, embed_dim=None, n_parameters=None)
        rows.append(rec)
    ck = pd.DataFrame(rows)
    ck.to_csv(os.path.join(rp, "checkpoint_manifest.tsv"), sep="\t", index=False)

    # task -> capability for both benchmarks
    pd.DataFrame([dict(benchmark="SinAEB", task=t, capability=c,
                       corpus=EA.CORPUS.get(t))
                  for t, c in EA.CAPABILITY.items()] +
                 [dict(benchmark="MAEB", task=t, capability=c, corpus=None)
                  for t, c in EA.MAEB_CAP.items()]).to_csv(
        os.path.join(rp, "task_capability_mapping.tsv"), sep="\t", index=False)

    # versions / environment
    env = {}
    for m in ["torch", "transformers", "mteb", "numpy", "scipy", "pandas",
              "sklearn", "soundfile", "soxr", "datasets"]:
        try:
            env[m] = getattr(__import__(m), "__version__", "unknown")
        except Exception:
            env[m] = "NOT INSTALLED"
    git = {}
    for k, cmd in (("commit", ["git", "rev-parse", "HEAD"]),
                   ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                   ("dirty_files", ["git", "status", "--porcelain"])):
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                 timeout=120).stdout.strip()
            git[k] = len(out.split("\n")) if k == "dirty_files" else out
        except Exception as e:
            git[k] = f"unavailable: {e}"
    try:
        patch = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True,
                               timeout=300).stdout
        open(os.path.join(rp, "local_changes.patch"), "w").write(patch)
        git["patch_bytes"] = len(patch)
        git["patch_note"] = ("tracked-file changes only; untracked files are NOT "
                             "in the patch - see untracked_files.txt")
    except Exception as e:
        git["patch"] = f"unavailable: {e}"
        missing.append("git diff patch")
    try:
        unt = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=120).stdout
        open(os.path.join(rp, "untracked_files.txt"), "w").write(unt)
    except Exception:
        missing.append("untracked file listing")

    json.dump(dict(packages=env, git=git,
                   python=sys.version.split()[0],
                   gpus="2x NVIDIA A30, driver 565.57.01",
                   run_window="2026-10-06 13:00 PDT to 2026-10-07 16:26 PDT"),
              open(os.path.join(rp, "environment.json"), "w"), indent=2)

    # split manifests / sample ids that exist
    for f in ["eval/addendum/classification_splits.tsv",
              "eval/analysis/S13_classification_split_ids.tsv",
              "eval/analysis/S14_rerank_query_ids.tsv",
              "eval/analysis/S7_matching_triplet_ids.tsv",
              "eval/analysis/S8_content_split_ids.tsv",
              "eval/subset.tsv",
              "eval/analysis/evaluation_config.json"]:
        if not cp(f, rp):
            missing.append(f)

    # ---------------------------------------------------------------- 08 logs
    lg = put(D["logs"])
    for f in ["expansion_gpu0.log", "expansion_gpu1.log",
              "family_bootstrap_final.log", "family_bootstrap_chain.log",
              "wave4_chain.log", "wave5_chain.log", "expansion_analysis.log"]:
        if not cp(f"logs/{f}", lg):
            missing.append(f"logs/{f}")

    # ------------------------------------------------------------- 09 figures
    fg = put(D["figs"])
    nf = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "eval/figures/*"))):
        if os.path.isfile(p):
            shutil.copy2(p, fg)
            nf += 1
    cp("eval/figures.py", fg)
    fsrc = put(os.path.join(D["figs"], "figure_source_data"))
    for f in ["V_layer_curves.tsv", "S1_full_matrix_long.tsv",
              "S6_maeb_capability_matched.tsv", "T_layer_transfer_classification.tsv",
              "U_classification_476_speakers.tsv", "expansion_task_corr.tsv"]:
        if not cp(f"eval/analysis/{f}", fsrc):
            missing.append(f"figure source {f}")

    # -------------------------------------------------- 10 layer study (6 enc)
    ly = put(D["layer"])
    for f in ["Q_layer_seeds_uniform.tsv", "Q2_per_seed_scores.tsv",
              "T_layer_transfer_classification.tsv",
              "U_classification_476_speakers.tsv", "V_layer_curves.tsv",
              "J_layer_transfer.tsv", "mfcc_matched_comparison.tsv",
              "mfcc_matched_per_seed.tsv", "S10_probe_convergence.tsv",
              "S11_strict_duration_subset.tsv", "S12_probe_reference_hard.tsv",
              "S15_convergence_all_conditions.tsv", "S16_cap_sensitivity.tsv",
              "S17_clustering_settings.tsv", "S19_aggregation_membership.tsv",
              "aggregate_ranks.tsv"]:
        if not cp(f"eval/analysis/{f}", ly):
            missing.append(f"layer study {f}")
    lysrc = put(os.path.join(D["layer"], "code"))
    for f in ["eval/extract.py", "eval/layer_curves.py", "eval/layer_seeds.py",
              "eval/layer_transfer.py", "eval/layer_transfer_clf.py",
              "eval/classification_476.py", "eval/mfcc_matched_comparison.py",
              "eval/supplementary.py"]:
        if not cp(f, lysrc):
            missing.append(f)

    json.dump(dict(
        encoders=["facebook/wav2vec2-xls-r-300m", "facebook/mms-300m",
                  "facebook/w2v-bert-2.0", "openai/whisper-large-v3",
                  "microsoft/wavlm-large", "facebook/hubert-large-ls960-ft"],
        n_encoders=6,
        scope=("These six are the ONLY encoders with per-layer extraction. The "
               "layer finding is established on them, not on the 45 checkpoints."),
        extraction=dict(layers="all hidden states incl. embedding output",
                        pooling="masked mean over non-padded frames",
                        dtype="bfloat16 forward, float32 accumulate, float16 on disk",
                        sample_rate=16000, subset_utterances=13219,
                        corpora={"openslr52": 11155, "openslr30": 2064}),
        cache_note=("eval/emb/*.npz per-layer embeddings are NOT included: "
                    "3.4 GB of derived arrays, regenerable with eval/extract.py")),
        open(os.path.join(ly, "layer_study_config.json"), "w"), indent=2)

    # ----------------------------------------------- 11 planned extension only
    ex = put(D["ext"])
    open(os.path.join(ex, "STATUS.md"), "w").write("""# Layer extension - PLANNED, NOT RUN

**Status as of packaging: not started. No results exist. Nothing is running.**

This folder holds a protocol only. It is included so the plan is on record, and
it must not be read as results.

## Correction on record

An earlier instruction to this packaging task said "do not interrupt the running
layer extension". There was no running extension. The extension was scoped only;
no extraction or probing was ever launched for it. The 45-checkpoint benchmark
run and the family-blocked bootstrap had both finished (16:26 on 2026-10-07)
before packaging began, and both GPUs were idle.

## Proposed protocol

Extend per-layer extraction from 6 encoders to 20, chosen as objective
contrasts rather than for breadth:

| pair / group | tests |
|---|---|
| hubert-large-ll60k vs hubert-large-ls960-ft (have) | SSL vs ASR-ft, same architecture |
| data2vec-audio-large vs -large-960h | same, second family |
| wav2vec2-base vs wav2vec2-base-960h | same, third family |
| wavlm-base-plus vs -sv vs -sd | SSL vs speaker-verification vs diarization |
| unispeech-sat-large, sew-d-mid, speecht5_asr, wav2vec2-conformer-rel-pos-large | new families |
| whisper-small, mms-1b-all | depth (12 vs 33 layers) and 48-layer scale |

Settings would be unchanged from the 6-encoder study: all hidden states, masked
mean pooling, 13,219 utterances, bfloat16 forward / float16 on disk.

## Measured and estimated cost

Extraction is measured from the existing logs: 3.3-13.2 min per encoder.
Probing cost is NOT measured and is the uncertain part; the 476-class GPU probe
runs at every layer and so scales with depth.

    extraction, 14 encoders   ~1.5 h one GPU / ~45 min across two   MEASURED basis
    probing, 4 analyses x 14  ~3-6 h                                ESTIMATE ONLY
    storage                   7.2 GB                                computed from configs

## Known risks, unresolved

- speecht5_asr is encoder-decoder; eval/extract.py special-cases Whisper only.
- sew-d downsamples internally, so its frame-to-mask mapping differs; its
  pooling mask needs checking or a padding confound returns silently.
- mms-1b-all at 48 layers x 1280 could exceed the probing estimate.

## Recommended first step

Pilot the WavLM triple (base-plus, -sv, -sd): ~15 min extraction, measures the
probing cost, and yields the single most informative figure - whether
speaker-objective finetuning moves the speaker peak later in the stack.
""")

    # checksums
    rows = []
    for dp, _, fs in os.walk(STAGE):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            rows.append(dict(path=os.path.relpath(p, STAGE),
                             bytes=os.path.getsize(p), sha256=h))
    man = pd.DataFrame(rows).sort_values("path")
    man.to_csv(os.path.join(STAGE, "MANIFEST_sha256.tsv"), sep="\t", index=False)

    print(f"\n  staged files: {len(man)}  "
          f"total {man.bytes.sum()/1e6:.1f} MB")
    print(f"  figures copied: {nf}")
    if missing:
        print(f"\n  NOT FOUND ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")
    json.dump(dict(counts=counts, staged_files=len(man),
                   staged_bytes=int(man.bytes.sum()), missing=missing),
              open(os.path.join(STAGE, "_build_report.json"), "w"), indent=2)
    return missing


if __name__ == "__main__":
    main()
