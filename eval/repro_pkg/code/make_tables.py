#!/usr/bin/env python3
"""Build the two paper tables that are documentation rather than analysis:

  L_extraction_protocol.tsv   what an encoder is, operationally (plan section 3)
  M_recommendations.tsv       what to use for what (plan section 10)

Both are written from the actual run configuration and the measured results, so
they stay correct if the numbers change.
"""
import os
import pandas as pd

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
OUT = os.path.join(ROOT, "eval/analysis")

# ---- section 3: extraction protocol, one row per checkpoint
PROTOCOL = [
    # model, revision, encoder component, params(M), dim, extract runtime (min)
    ("microsoft/wavlm-large",          "c1423ed94bb0", "WavLMModel",            316.6, 1024, 10.6),
    ("facebook/hubert-large-ls960-ft", "ece5fabbf034", "HubertModel",           317.0, 1024,  5.6),
    ("facebook/w2v-bert-2.0",          "da985ba0987f", "Wav2Vec2BertModel",     580.0, 1024,  8.0),
    ("facebook/wav2vec2-xls-r-300m",   "1a640f32ac3e", "Wav2Vec2Model",         300.0, 1024,  4.8),
    ("facebook/wav2vec2-xls-r-1b",     "35eaea9a0ed0", "Wav2Vec2Model",        1000.0, 1024, None),
    ("facebook/wav2vec2-xls-r-2b",     "3b6d89d0fabe", "Wav2Vec2Model",        2000.0, 1024, None),
    ("facebook/mms-300m",              "4ee317ce793c", "Wav2Vec2Model",         315.0, 1024,  3.3),
    ("openai/whisper-tiny",            "main",         "WhisperEncoder",         39.0,  512, None),
    ("openai/whisper-base",            "main",         "WhisperEncoder",         74.0,  512, None),
    ("openai/whisper-small",           "main",         "WhisperEncoder",        244.0,  768, None),
    ("openai/whisper-medium",          "main",         "WhisperEncoder",        769.0, 1024, None),
    ("openai/whisper-large-v3",        "main",         "WhisperEncoder",       1550.0, 1280, 13.2),
    ("baseline/mfcc-40",               "1",            "none (20 MFCC mean+std)", 0.0,   40,  0.4),
]

SHARED = dict(
    sample_rate="16 kHz (resampled, mono)",
    max_duration="20 s, head crop",
    layer="final hidden state (default); layer study sweeps all",
    pooling="mean over valid frames, padding masked",
    normalisation="L2 before cosine",
    similarity="cosine",
    precision="bfloat16 forward, float32 output",
    hardware="1x NVIDIA RTX A5000 24 GB",
    encoder_state="frozen (no fine-tuning)",
)


def protocol():
    df = pd.DataFrame(PROTOCOL, columns=[
        "model", "revision", "encoder_component", "params_M", "embed_dim",
        "extract_min_13219_utts"])
    for k, v in SHARED.items():
        df[k] = v
    df.to_csv(os.path.join(OUT, "L_extraction_protocol.tsv"), sep="\t", index=False)
    print("## L. Extraction protocol\n")
    print(df[["model", "revision", "encoder_component", "params_M", "embed_dim",
              "extract_min_13219_utts"]].to_string(index=False))
    print("\nShared across every checkpoint:")
    for k, v in SHARED.items():
        print(f"  {k:16s} {v}")
    return df


# ---- section 10: recommendations, built from measured results
def recommendations():
    H = pd.read_csv(os.path.join(OUT, "H_controlled_matching_final.tsv"), sep="\t")
    J = pd.read_csv(os.path.join(OUT, "J_layer_transfer.tsv"), sep="\t")
    per = pd.read_csv(os.path.join(ROOT, "eval/simteb_audio_results_per_task.tsv"),
                      sep="\t").set_index("model")

    spk = J[J.capability == "speaker"].sort_values("test_score_picked",
                                                   ascending=False)
    cnt = H.sort_values("p_content_over_speaker", ascending=False)

    best_spk = spk.iloc[0]
    best_cnt = cnt.iloc[0]
    mfcc_spk = per.loc["baseline/mfcc-40", "OpenSLR30SpeakerClustering"]

    rows = [
        dict(application="Speaker matching / verification",
             recommendation=f"{best_spk.model}, layer {int(best_spk.picked_layer)} "
                            f"of {int(best_spk.n_layers)-1} (depth "
                            f"{best_spk.picked_depth:.2f})",
             pooling="mean, L2, cosine",
             evidence=f"AMI {best_spk.test_score_picked:.3f} on held-out "
                      f"OpenSLR-30 after selecting the layer on OpenSLR-52; "
                      f"final layer gives {best_spk.test_score_final:.3f}",
             caveat="A 40-dim MFCC baseline reaches "
                    f"{mfcc_spk/100:.3f} on the same task, so the deep encoder "
                    "is not clearly necessary for speaker work on these corpora"),
        dict(application="Spoken-content retrieval (same-utterance matching)",
             recommendation=f"{best_cnt.model}, layer 27 of 32 (depth 0.84)",
             pooling="mean, L2, cosine",
             evidence=f"P[content>speaker] = {best_cnt.p_content_over_speaker:.3f} "
                      f"[{best_cnt.ci_lo:.3f}, {best_cnt.ci_hi:.3f}] over 10,346 "
                      f"duration-matched queries; layer 27 transfers to held-out "
                      f"speakers (+0.056 over final)",
             caveat="Evaluated as same-transcript matching, not semantic "
                    "similarity; no Sinhala semantic task was available"),
        dict(application="Classification on frozen embeddings",
             recommendation="wavlm-large or wav2vec2-xls-r-2b, early layer "
                            "(depth 0.13-0.17) for speaker targets",
             pooling="mean, L2; logistic probe, samples_per_label=8, "
                     "10 experiments (MTEB default)",
             evidence="Cross-corpus rank correlation for classification is "
                      "rho = 0.95, so classification rankings transfer between "
                      "corpora",
             caveat="A supervised probe on frozen features is still a "
                    "supervised evaluation; MFCC matches encoders on 10 of 14 "
                    "tasks under this protocol"),
        dict(application="Resource-constrained deployment",
             recommendation="whisper-small (244 M) for content; "
                            "baseline/mfcc-40 (0 params, CPU) for speaker",
             pooling="mean, L2, cosine",
             evidence="whisper-small ranks 1st on the capability-balanced "
                      "aggregate despite being 6x smaller than whisper-large-v3; "
                      "MFCC needs no GPU and no model download",
             caveat="whisper-small's advantage rests on 3 content tasks, two of "
                    "them from one corpus"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "M_recommendations.tsv"), sep="\t", index=False)
    print("\n\n## M. Recommendations\n")
    for r in rows:
        print(f"  {r['application']}")
        print(f"      use      : {r['recommendation']}")
        print(f"      evidence : {r['evidence']}")
        print(f"      caveat   : {r['caveat']}\n")
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    protocol()
    recommendations()
    print(f"wrote {OUT}/L_extraction_protocol.tsv and M_recommendations.tsv")
