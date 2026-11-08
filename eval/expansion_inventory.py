#!/usr/bin/env python3
"""The checkpoint inventory for the expansion, ordered so that stopping early
still leaves a usable, balanced sample.

The ordering principle matters more than the list. Two rules:

  MODEL-COMPLETE, NOT TASK-COMPLETE
      Every task of both benchmarks is finished for one checkpoint before the
      next begins. Stopping after N checkpoints leaves N usable rows, not 45
      half-filled ones. A half-finished checkpoint contributes nothing to a
      cross-benchmark correlation, so the unit of progress has to be the
      checkpoint.

  BREADTH FIRST, DEPTH SECOND
      The list is laid out in waves. Wave 1 takes one checkpoint from each
      family, wave 2 takes a second size from each, and so on. Stopping after
      wave 2 gives a sample spread across families rather than six sizes of
      Whisper. Size variants within a family are not independent evidence, so
      front-loading them would inflate the count without improving it.

Each entry records the extraction rule, because the comparison is only valid if
both benchmarks see the same embedding for a checkpoint.
"""
import json
import os

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"

# wave, hf id, family, approximate encoder params (M), objective, notes
CANDIDATES = [
    # ---- wave 0: already evaluated, kept as the base
    (0, "microsoft/wavlm-large", "wavlm", 316, "self-supervised", "done"),
    (0, "facebook/hubert-large-ls960-ft", "hubert", 315, "asr-finetuned", "done"),
    (0, "facebook/w2v-bert-2.0", "w2v-bert", 580, "self-supervised", "done"),
    (0, "facebook/wav2vec2-xls-r-300m", "xls-r", 315, "self-supervised", "done"),
    (0, "facebook/wav2vec2-xls-r-1b", "xls-r", 963, "self-supervised", "done"),
    (0, "facebook/wav2vec2-xls-r-2b", "xls-r", 2162, "self-supervised", "done"),
    (0, "facebook/mms-300m", "mms", 315, "self-supervised", "done"),
    (0, "openai/whisper-tiny", "whisper", 8, "asr-supervised", "done"),
    (0, "openai/whisper-base", "whisper", 21, "asr-supervised", "done"),
    (0, "openai/whisper-small", "whisper", 88, "asr-supervised", "done"),
    (0, "openai/whisper-medium", "whisper", 307, "asr-supervised", "done"),
    (0, "openai/whisper-large-v3", "whisper", 637, "asr-supervised", "done"),

    # ---- wave 1: one per NEW family. maximum breadth per unit of compute
    (1, "microsoft/wavlm-base-plus", "wavlm", 94, "self-supervised", "new size"),
    (1, "facebook/hubert-base-ls960", "hubert", 94, "self-supervised",
     "NOT fine-tuned - pairs with hubert-large-ls960-ft"),
    (1, "facebook/wav2vec2-base", "wav2vec2", 94, "self-supervised", "new family"),
    (1, "facebook/mms-1b-all", "mms", 965, "asr-finetuned", "new size + objective"),
    (1, "microsoft/unispeech-sat-large", "unispeech", 316, "self-supervised",
     "new family, speaker-aware pretraining"),
    (1, "facebook/data2vec-audio-large", "data2vec", 313, "self-supervised", "new family"),
    (1, "openai/whisper-large-v2", "whisper", 637, "asr-supervised", "v2 vs v3"),
    (1, "facebook/wav2vec2-large-robust", "wav2vec2", 315, "self-supervised",
     "replaces the NeMo parakeet entry, which the transformers wrapper cannot load"),

    # ---- wave 2: second size in each family
    (2, "microsoft/wavlm-base", "wavlm", 94, "self-supervised", ""),
    (2, "facebook/wav2vec2-large-960h", "wav2vec2", 315, "asr-finetuned", ""),
    (2, "facebook/data2vec-audio-base", "data2vec", 93, "self-supervised", ""),
    (2, "microsoft/unispeech-sat-base-plus", "unispeech", 94, "self-supervised", ""),
    (2, "facebook/mms-1b-fl102", "mms", 965, "asr-finetuned", ""),
    (2, "facebook/wav2vec2-large-xlsr-53", "xls-r", 315, "self-supervised",
     "the earlier XLSR generation"),
    (2, "openai/whisper-small.en", "whisper", 88, "asr-supervised",
     "English-only, pairs with whisper-small"),
    (2, "facebook/hubert-xlarge-ll60k", "hubert", 964, "self-supervised", ""),

    # ---- wave 3: remaining variants, lowest marginal information
    (3, "microsoft/wavlm-base-sv", "wavlm", 94, "speaker-verification-finetuned",
     "directly relevant to the speaker capability"),
    (3, "facebook/wav2vec2-base-960h", "wav2vec2", 94, "asr-finetuned", ""),
    (3, "facebook/hubert-large-ll60k", "hubert", 316, "self-supervised",
     "not fine-tuned, pairs with the ls960-ft variant"),
    (3, "microsoft/unispeech-sat-base", "unispeech", 94, "self-supervised", ""),
    (3, "facebook/data2vec-audio-base-960h", "data2vec", 93, "asr-finetuned", ""),
    (3, "openai/whisper-medium.en", "whisper", 307, "asr-supervised", ""),
    (3, "openai/whisper-base.en", "whisper", 21, "asr-supervised", ""),
    (3, "openai/whisper-tiny.en", "whisper", 8, "asr-supervised", ""),
    (3, "facebook/wav2vec2-xls-r-300m-21-to-en", "xls-r", 315, "translation-finetuned",
     "sequence-to-sequence, encoder only"),
    (3, "microsoft/wavlm-base-plus-sv", "wavlm", 94, "speaker-verification-finetuned",
     "replaces wavlm-large-sv, which does not exist on the Hub"),

    # ---- wave 4: reaching 45. Chosen for NEW families and objectives rather
    # than further size variants, which add count without adding independent
    # evidence. All are already in MTEB's registry, so no wrapper work.
    (4, "asapp/sew-d-base-plus-400k-ft-ls100h", "sew-d", 95, "asr-finetuned",
     "NEW FAMILY: squeezed wav2vec2 variant"),
    (4, "asapp/sew-d-mid-400k-ft-ls100h", "sew-d", 139, "asr-finetuned",
     "second SEW-D size"),
    (4, "microsoft/speecht5_asr", "speecht5", 152, "asr-finetuned",
     "NEW FAMILY: encoder-decoder speech/text"),
    (4, "speechbrain/m-ctc-t-large", "m-ctc-t", 1059, "asr-finetuned",
     "NEW FAMILY: massively multilingual CTC transformer"),
    (4, "microsoft/wavlm-base-plus-sd", "wavlm", 95, "diarization-finetuned",
     "NEW OBJECTIVE: speaker diarization, directly relevant to the speaker capability"),
    (4, "facebook/data2vec-audio-large-960h", "data2vec", 313, "asr-finetuned",
     "data2vec with an ASR objective, pairs with the self-supervised large"),
    (4, "openai/whisper-large-v3-turbo", "whisper", 809, "asr-supervised",
     "distilled-decoder variant; encoder differs from large-v3"),
]

EXTRACTION = {
    "sample_rate_hz": 16000, "channels": "mono (averaged)",
    "max_duration_s": 20.0, "crop": "head",
    "layer": "final hidden state",
    "pooling": "mean over valid frames, padding masked",
    "normalisation": "L2 before cosine", "similarity": "cosine",
    "precision": "bfloat16 forward, float32 output, float16 on disk",
    "wrapper": "simteb_audio.models.extra_audio_models / mteb 2.21.10",
    "note": "identical for SinAEB and MAEB; a checkpoint whose embedding "
            "differs between benchmarks cannot enter the correlation",
}


def main():
    rows = [dict(wave=w, model=m, family=f, encoder_params_m=p, objective=o, note=n)
            for w, m, f, p, o, n in CANDIDATES]
    import pandas as pd
    d = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/expansion_inventory.tsv")
    d.to_csv(out, sep="\t", index=False)

    print(f"  total candidates: {len(d)}  "
          f"(wave 0 already done: {int((d.wave == 0).sum())})")
    print()
    cum = int((d.wave == 0).sum())
    for w in (1, 2, 3, 4):
        n = int((d.wave == w).sum())
        cum += n
        fams = d[d.wave <= w].family.nunique()
        print(f"  after wave {w}: +{n:2d} checkpoints -> {cum:2d} total, "
              f"{fams} families, {d[d.wave <= w].objective.nunique()} objectives")
    print()
    print("  families represented:", ", ".join(sorted(d.family.unique())))
    print("  objectives:", ", ".join(sorted(d.objective.unique())))
    print()
    print("  flagged for compatibility checking:")
    for _, r in d[d.note.str.contains("UNVERIFIED", na=False)].iterrows():
        print(f"    {r.model}  ({r.note})")

    with open(os.path.join(ROOT, "eval/analysis/extraction_rule.json"), "w") as f:
        json.dump(EXTRACTION, f, indent=2)
    print(f"\nwrote {out}")
    print(f"wrote {ROOT}/eval/analysis/extraction_rule.json")


if __name__ == "__main__":
    main()
