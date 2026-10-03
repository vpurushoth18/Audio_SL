#!/usr/bin/env python3
"""Does batching change an embedding? If the mask is right, it cannot.

Re-encodes a sample of clips ONE AT A TIME, where no padding exists, and
compares against the cached embedding produced in batches of 16 alongside clips
of other lengths. A correct mask makes the two agree to numerical precision; a
leaking mask makes short clips differ most, because they carry the most padding.

Sampled deliberately: the shortest clips, the longest, and a random middle, so a
length-dependent failure cannot hide in the average.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join("/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL", "eval"))
from extract import MODELS, load_audio, SR          # noqa: E402

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
N_PER_BUCKET = 12


@torch.no_grad()
def encode_one(model, fe, wave, is_whisper, device, dtype):
    """One clip, no batch, therefore no padding."""
    if is_whisper:
        feats = fe([wave], sampling_rate=SR, return_tensors="pt")
        out = model(feats.input_features.to(device, dtype), output_hidden_states=True)
        nfr = max(1, min(1500, int(len(wave) / SR * 50)))
        hs = torch.stack(out.hidden_states)[:, :, :nfr, :]
    else:
        feats = fe([wave], sampling_rate=SR, return_tensors="pt")
        key = "input_features" if "input_features" in feats else "input_values"
        out = model(feats[key].to(device, dtype), output_hidden_states=True)
        hs = torch.stack(out.hidden_states)
    return hs.mean(2)[:, 0, :].float().cpu().numpy()


def main():
    tags = sys.argv[1:] or ["wavlm", "whisper"]
    device, dtype = "cuda:0", torch.bfloat16
    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3,
                      dtype={"speaker_id": str})
    s = sub.sort_values("duration_s")
    pick = pd.concat([s.head(N_PER_BUCKET), s.tail(N_PER_BUCKET),
                      s.sample(N_PER_BUCKET, random_state=0)]).drop_duplicates("utt_id")
    bucket = (["shortest"] * N_PER_BUCKET + ["longest"] * N_PER_BUCKET
              + ["random"] * N_PER_BUCKET)[:len(pick)]

    rows = []
    for tag in tags:
        repo = MODELS[tag]
        from transformers import AutoModel, AutoFeatureExtractor
        is_whisper = "whisper" in repo
        fe = AutoFeatureExtractor.from_pretrained(repo)
        if is_whisper:
            from transformers import WhisperModel
            model = WhisperModel.from_pretrained(repo, torch_dtype=dtype).encoder
        else:
            model = AutoModel.from_pretrained(repo, torch_dtype=dtype)
        model.to(device).eval()

        z = np.load(os.path.join(ROOT, f"eval/emb/{tag}.npz"), allow_pickle=True)
        cached = {u: i for i, u in enumerate(z["utt"].astype(str))}
        emb = z["emb"]

        for (_, r), b in zip(pick.iterrows(), bucket):
            if r.utt_id not in cached:
                continue
            single = encode_one(model, fe, load_audio(r.path, 20.0), is_whisper,
                                device, dtype)
            batched = emb[:, cached[r.utt_id], :].astype(np.float32)
            fl_s, fl_b = single[-1], batched[-1]
            cos = float(fl_s @ fl_b / max(np.linalg.norm(fl_s) * np.linalg.norm(fl_b), 1e-9))
            rel = float(np.linalg.norm(fl_s - fl_b) / max(np.linalg.norm(fl_b), 1e-9))
            rows.append(dict(model=repo, tag=tag, bucket=b, utt=r.utt_id,
                             duration_s=r.duration_s, cosine_single_vs_batched=cos,
                             relative_l2_error=rel))
        del model, z, emb
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "eval/analysis/R2_padding_batch_invariance.tsv")
    df.to_csv(out, sep="\t", index=False)
    print("\n## R2. Batched vs single-clip encoding (final layer)\n")
    print(df.groupby(["tag", "bucket"]).agg(
        n=("utt", "size"), mean_duration=("duration_s", "mean"),
        min_cosine=("cosine_single_vs_batched", "min"),
        mean_cosine=("cosine_single_vs_batched", "mean"),
        max_rel_error=("relative_l2_error", "max")).to_string(
            float_format=lambda x: f"{x:.5f}"))
    print("\n  cosine at 1.000 for every bucket means batch composition does not\n"
          "  change the embedding, so the padding mask is applied correctly.\n"
          "  A leak would show up first in the 'shortest' bucket, which carries\n"
          "  the most padding when batched with long clips.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
