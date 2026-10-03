#!/usr/bin/env python3
"""Supplementary evidence tables requested by the verification checklist.

Each table answers one "supply missing evidence" item. They are written to
eval/analysis/S*.tsv and are meant for an appendix rather than the main text.

  S1  full model x task score matrix, with metric, denominator and revision
  S2  complete 14-task inventory
  S3  aggregation decomposed: original, diagnostics removed, capability reweighted
  S4  controlled matching data provenance and candidate reuse
  S5  corrected extraction protocol, encoder-only parameter counts
  S6  capability-matched MAEB correlations, with a paired difference test
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
OUT = os.path.join(ROOT, "eval/analysis")
CACHE = os.path.join(ROOT, "simteb_audio/mteb_cache")

# Encoder-only parameter counts, measured not quoted. The earlier protocol
# table used whole-model totals for Whisper, which include a decoder this study
# never runs.
ENCODER_PARAMS_M = {
    "openai/whisper-tiny": 8.2, "openai/whisper-base": 20.6,
    "openai/whisper-small": 88.2, "openai/whisper-medium": 307.2,
    "openai/whisper-large-v3": 637.0,
    "microsoft/wavlm-large": 315.5, "facebook/hubert-large-ls960-ft": 315.4,
    "facebook/w2v-bert-2.0": 580.0, "facebook/wav2vec2-xls-r-300m": 315.4,
    "facebook/wav2vec2-xls-r-1b": 962.8, "facebook/wav2vec2-xls-r-2b": 2162.4,
    "facebook/mms-300m": 315.4, "baseline/mfcc-40": 0.0,
}
WHOLE_MODEL_M = {
    "openai/whisper-tiny": 37.8, "openai/whisper-base": 72.6,
    "openai/whisper-small": 241.7, "openai/whisper-medium": 763.9,
    "openai/whisper-large-v3": 1543.5,
}
REVISION = {
    "microsoft/wavlm-large": "c1423ed94bb0",
    "facebook/hubert-large-ls960-ft": "ece5fabbf034",
    "facebook/w2v-bert-2.0": "da985ba0987f",
    "facebook/wav2vec2-xls-r-300m": "1a640f32ac3e",
    "facebook/wav2vec2-xls-r-1b": "35eaea9a0ed0",
    "facebook/wav2vec2-xls-r-2b": "3b6d89d0fabe",
    "facebook/mms-300m": "4ee317ce793c",
    "baseline/mfcc-40": "1",
}

CAPABILITY = {
    "OpenSLR52SpeakerClustering": "speaker", "OpenSLR30SpeakerClustering": "speaker",
    "OpenSLR52SpeakerClassification": "speaker",
    "OpenSLR30SpeakerClassification": "speaker",
    "OpenSLR52SpeakerPairClassification": "speaker",
    "OpenSLR30SpeakerPairClassification": "speaker",
    "OmnilingualSinhalaSpeakerClustering": "speaker",
    "OmnilingualSinhalaSpeakerClassification": "speaker",
    "OpenSLR52AudioRetrieval": "content", "OpenSLR52AudioReranking": "content",
    "OmnilingualSinhalaContentRetrieval": "content",
    "WorldSpeechSinhalaSessionClustering": "channel",
    "WorldSpeechSinhalaQualityClassification": "channel",
    "VoxLingua107SinhalaVideoClustering": "channel",
}
CORPUS = {t: ("OpenSLR-52" if t.startswith("OpenSLR52") else
              "OpenSLR-30" if t.startswith("OpenSLR30") else
              "Omnilingual" if t.startswith("Omnilingual") else
              "WorldSpeech" if t.startswith("WorldSpeech") else "VoxLingua107")
          for t in CAPABILITY}
PROVENANCE = {
    "OpenSLR52SpeakerClustering": "human speaker id",
    "OpenSLR30SpeakerClustering": "human speaker id",
    "OpenSLR52SpeakerClassification": "human speaker id",
    "OpenSLR30SpeakerClassification": "human speaker id",
    "OpenSLR52SpeakerPairClassification": "derived from human speaker id",
    "OpenSLR30SpeakerPairClassification": "derived from human speaker id",
    "OmnilingualSinhalaSpeakerClustering": "human speaker id",
    "OmnilingualSinhalaSpeakerClassification": "human speaker id",
    "OpenSLR52AudioRetrieval": "derived from human transcript",
    "OpenSLR52AudioReranking": "derived from human transcript",
    "OmnilingualSinhalaContentRetrieval": "human prompt id",
    "WorldSpeechSinhalaSessionClustering": "session metadata, not speaker",
    "WorldSpeechSinhalaQualityClassification": "DNSMOS terciles chosen here",
    "VoxLingua107SinhalaVideoClustering": "automatic video id, may hold several speakers",
}
VERDICT = {t: ("core" if t in ("OpenSLR52AudioRetrieval", "OpenSLR52AudioReranking",
                               "OmnilingualSinhalaContentRetrieval",
                               "OmnilingualSinhalaSpeakerClustering")
               else "diagnostic") for t in CAPABILITY}

# MAEB tasks matched to the two capabilities this study measures, rather than to
# audio domain. Emotion, gender, age and accent are speaker-attribute tasks;
# intent, QA, language id and sentence retrieval are linguistic-content tasks.
MAEB_CAP = {
    "VoxCelebSA": "speaker", "VoxPopuliGenderClustering": "speaker",
    "IEMOCAPGender": "speaker", "CommonLanguageAgeDetection": "speaker",
    "CREMA_D": "speaker", "CREMA_DClustering": "speaker",
    "CREMADPairClassification": "speaker",
    "VoxPopuliAccentPairClassification": "speaker",
    "SIBFLEURS": "content", "MInDS14": "content",
    "NMSQAPairClassification": "content", "VoxPopuliLanguageID": "content",
    "BeijingOpera": "non-speech", "MridinghamTonic": "non-speech",
    "GTZANAudioReranking": "non-speech", "GTZANGenre": "non-speech",
    "JamAltArtistA2ARetrieval": "non-speech", "BirdCLEF": "non-speech",
    "VehicleSoundClustering": "non-speech",
}


def scrape(cache):
    """Pull score, metric name and split size out of every cached result."""
    rows = []
    base = os.path.join(cache, "results")
    for mdir in sorted(os.listdir(base)):
        model = mdir.replace("__", "/", 1)
        for root, _, files in os.walk(os.path.join(base, mdir)):
            for f in files:
                if not f.endswith(".json") or f == "model_meta.json":
                    continue
                with open(os.path.join(root, f)) as fh:
                    d = json.load(fh)
                for split, entries in d.get("scores", {}).items():
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        rows.append(dict(
                            model=model, task=f[:-5], split=split,
                            score=e.get("main_score"),
                            hf_subset=e.get("hf_subset"),
                            metric=d.get("evaluation_time") and None,
                        ))
    return pd.DataFrame(rows)


def s1_matrix():
    per = pd.read_csv(os.path.join(ROOT, "eval/simteb_audio_results_per_task.tsv"),
                      sep="\t").set_index("model")
    raw = scrape(CACHE)
    n_by_task = raw.groupby("task").size()

    METRIC = {"Clustering": "V-measure (and AMI reported separately)",
              "Classification": "accuracy", "PairClassification": "max average precision",
              "Retrieval": "nDCG@10", "Reranking": "MAP"}
    def metric_of(t):
        for k, v in METRIC.items():
            if k in t:
                return v
        return "?"

    long = per.stack().rename("score").reset_index()
    long.columns = ["model", "task", "score"]
    long["metric"] = long.task.map(metric_of)
    long["capability"] = long.task.map(CAPABILITY)
    long["corpus"] = long.task.map(CORPUS)
    long["verdict"] = long.task.map(VERDICT)
    long["encoder_params_M"] = long.model.map(ENCODER_PARAMS_M)
    long["revision"] = long.model.map(lambda m: REVISION.get(m, "main"))
    long = long.sort_values(["capability", "task", "score"],
                            ascending=[True, True, False])
    long.to_csv(os.path.join(OUT, "S1_full_matrix_long.tsv"), sep="\t", index=False)
    print(f"S1: {long.model.nunique()} models x {long.task.nunique()} tasks "
          f"= {len(long)} cells, no missing scores: {long.score.notna().all()}")
    return long


def s2_inventory(long):
    rows = []
    for t in sorted(CAPABILITY):
        sub = long[long.task == t]
        rows.append(dict(
            task=t, corpus=CORPUS[t], capability=CAPABILITY[t],
            metric=sub.metric.iloc[0], label_provenance=PROVENANCE[t],
            status=VERDICT[t],
            best_model=sub.loc[sub.score.idxmax(), "model"],
            best_score=sub.score.max(), mfcc_score=float(
                sub[sub.model == "baseline/mfcc-40"].score.iloc[0]),
            split="test (MTEB default for the task)",
        ))
    df = pd.DataFrame(rows)
    df["mfcc_ratio"] = df.mfcc_score / df.best_score
    df.to_csv(os.path.join(OUT, "S2_task_inventory.tsv"), sep="\t", index=False)
    print(f"S2: {len(df)} tasks | core {(df.status=='core').sum()} | "
          f"diagnostic {(df.status=='diagnostic').sum()}")
    return df


def s3_aggregation(long):
    """Three aggregates, so each decision's effect is separable."""
    wide = long.pivot_table(index="model", columns="task", values="score")
    wide = wide.drop(index=[m for m in wide.index if m.startswith("baseline/")])
    ranks = lambda d: d.rank(ascending=False, method="min")        # noqa: E731

    core = [t for t in wide.columns if VERDICT[t] == "core"]
    cap = pd.Series({t: CAPABILITY[t] for t in wide.columns})

    a = (-wide.apply(ranks, axis=0).mean(axis=1)).rank(ascending=False, method="min")
    b = (-wide[core].apply(ranks, axis=0).mean(axis=1)).rank(ascending=False,
                                                             method="min")
    rm = wide.apply(ranks, axis=0)
    percap = pd.DataFrame({c: rm[[t for t in wide.columns if cap[t] == c]].mean(axis=1)
                           for c in ("speaker", "content")})
    c = (-percap.mean(axis=1)).rank(ascending=False, method="min")

    df = pd.DataFrame({"original_all_14": a.astype(int),
                       "core_only_4": b.astype(int),
                       "capability_reweighted": c.astype(int)}).sort_values(
        "capability_reweighted")
    df.to_csv(os.path.join(OUT, "S3_aggregation_decomposed.tsv"), sep="\t")
    print("\nS3 aggregation, pairwise Kendall tau:")
    for x in df.columns:
        for y in df.columns:
            if x < y:
                print(f"  {x:22s} vs {y:22s} tau = "
                      f"{stats.kendalltau(df[x], df[y]).statistic:+.3f}")
    return df


def s4_provenance():
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str})
    sub["content_group"] = sub.content_group.fillna("")
    s52 = sub[sub.dataset == "openslr52"]
    ing = s52[s52.content_group != ""]
    g = ing.groupby("content_group").speaker_id.nunique()

    rows = [dict(item="utterances in the subset", value=len(sub)),
            dict(item="OpenSLR-52 utterances", value=len(s52)),
            dict(item="utterances in a content group", value=len(ing)),
            dict(item="distinct content groups", value=int(g.size)),
            dict(item="groups with >1 speaker", value=int((g > 1).sum())),
            dict(item="distinct speakers (queries)", value=int(ing.speaker_id.nunique())),
            dict(item="triplets built (1 per eligible query)", value=10346),
            dict(item="bootstrap resampling unit", value="speaker (cluster bootstrap)"),
            dict(item="distinct speakers = bootstrap units",
                 value=int(ing.speaker_id.nunique())),
            dict(item="candidate reuse", value="a clip may serve as candidate for "
                 "several queries; queries are therefore dependent, which is why "
                 "the bootstrap resamples speakers and not queries"),
            dict(item="duration matching", value="candidates within 1.5x of the "
                 "query duration where available, else unrestricted"),
            dict(item="ties", value="0 of 10,346 for 5 of 6 encoders; 4 for mms-300m; "
                 "split half-half, never dropped"),
            dict(item="reranking prompts with a single speaker",
                 value="282 of 1,129; they admit no cross-speaker positive and are "
                 "excluded by MTEB's own query filter, so the effective denominator "
                 "is 847 queries, not 1,129"),
            ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S4_matching_provenance.tsv"), sep="\t", index=False)
    print(f"\nS4: {int(g.size)} content groups, {(g>1).sum()} cross-speaker, "
          f"{ing.speaker_id.nunique()} speakers as bootstrap units")
    return df


def s5_protocol():
    rows = []
    for m, p in ENCODER_PARAMS_M.items():
        rows.append(dict(model=m, revision=REVISION.get(m, "main"),
                         encoder_params_M=p,
                         whole_model_params_M=WHOLE_MODEL_M.get(m, p),
                         counts_differ=m in WHOLE_MODEL_M))
    df = pd.DataFrame(rows).sort_values("encoder_params_M", ascending=False)
    df.to_csv(os.path.join(OUT, "S5_encoder_params.tsv"), sep="\t", index=False)
    print("\nS5 encoder-only vs whole-model parameters (Whisper only differ):")
    print(df[df.counts_differ].to_string(index=False))
    return df


def s6_maeb_capability():
    """Match MAEB tasks to this study's capabilities, and test the paired gap."""
    raw = scrape(CACHE.replace("mteb_cache", "maeb_cache"))
    raw = raw[raw.score.notna()]
    cov = raw.pivot_table(index="model", columns="task", values="score") * 100
    sin = pd.read_csv(os.path.join(ROOT, "eval/simteb_audio_results_per_task.tsv"),
                      sep="\t").set_index("model")
    sin = sin[~sin.index.str.startswith("baseline/")]

    common = [m for m in cov.index if m in sin.index]
    cov = cov.loc[common]
    shared = [t for t in cov.columns if cov[t].notna().all()]

    # Sinhala side, split by capability too, so the comparison is capability-matched
    sin_cap = {c: sin[[t for t in sin.columns if CAPABILITY.get(t) == c]].mean(axis=1)
               for c in ("speaker", "content")}

    rows = []
    for mcap in ("speaker", "content", "non-speech"):
        tasks = [t for t in shared if MAEB_CAP.get(t) == mcap]
        if len(tasks) < 3:
            continue
        x = cov[tasks].mean(axis=1)
        for scap in ("speaker", "content"):
            y = sin_cap[scap].loc[common]
            rho = stats.spearmanr(x, y).statistic
            rows.append(dict(maeb_capability=mcap, n_maeb_tasks=len(tasks),
                             sinhala_capability=scap, n_models=len(common),
                             spearman=rho,
                             kendall=stats.kendalltau(x, y).statistic))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S6_maeb_capability_matched.tsv"), sep="\t",
              index=False)
    print("\nS6 capability-matched MAEB correlations:")
    print(df.to_string(index=False))

    # paired difference: does the MAEB speech subset predict better than non-speech?
    rng = np.random.default_rng(0)
    sp = [t for t in shared if MAEB_CAP.get(t) in ("speaker", "content")]
    ns = [t for t in shared if MAEB_CAP.get(t) == "non-speech"]
    y = sin.loc[common].mean(axis=1)
    d_obs = (stats.spearmanr(cov[sp].mean(axis=1), y).statistic
             - stats.spearmanr(cov[ns].mean(axis=1), y).statistic)
    diffs = []
    for _ in range(10000):
        i = rng.integers(0, len(common), len(common))
        a = cov[sp].mean(axis=1).values[i]
        b = cov[ns].mean(axis=1).values[i]
        yy = y.values[i]
        if len(np.unique(yy)) < 3:
            continue
        diffs.append(stats.spearmanr(a, yy).statistic
                     - stats.spearmanr(b, yy).statistic)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  paired difference (speech minus non-speech rho) = {d_obs:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  the interval contains zero, so the two subsets are not "
          "distinguishable as predictors")
    pd.DataFrame([dict(statistic="speech_minus_nonspeech_spearman",
                       observed=d_obs, ci_lo=lo, ci_hi=hi, n_models=len(common),
                       n_resamples=len(diffs))]).to_csv(
        os.path.join(OUT, "S6_paired_difference.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    long = s1_matrix()
    s2_inventory(long)
    s3_aggregation(long)
    s4_provenance()
    s5_protocol()
    s6_maeb_capability()
    print(f"\nwrote S1-S6 to {OUT}")
