#!/usr/bin/env python3
"""Embed the evaluation subset with a frozen speech encoder, all layers, mean-pooled.

Writes <out>/<tag>.npz holding:
  emb   float16 [n_layers, n_utts, dim]   masked mean-pooled per layer
  utt   the utterance ids, aligned to axis 1
Padding is excluded from the pooling via the encoder's own downsampled mask,
otherwise short clips in a batch get their means dragged toward the pad vector.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, soundfile as sf, soxr, torch

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
SR = 16000

MODELS = {
    "xlsr300m":  "facebook/wav2vec2-xls-r-300m",
    "mms300m":   "facebook/mms-300m",
    "w2vbert2":  "facebook/w2v-bert-2.0",
    "whisper":   "openai/whisper-large-v3",
}


def load_audio(path, max_s):
    x, sr = sf.read(os.path.join(ROOT, path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    if max_s:
        x = x[: int(max_s * SR)]
    return x


@torch.no_grad()
def run(tag, repo, sub, out_dir, batch, max_s, device, dtype):
    from transformers import AutoModel, AutoFeatureExtractor
    is_whisper = "whisper" in repo

    fe = AutoFeatureExtractor.from_pretrained(repo)
    if is_whisper:
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained(repo, torch_dtype=dtype).encoder
    else:
        model = AutoModel.from_pretrained(repo, torch_dtype=dtype)
    model.to(device).eval()

    embs, utts = [], []
    t0 = time.time()
    for i in range(0, len(sub), batch):
        chunk = sub.iloc[i : i + batch]
        waves = [load_audio(p, max_s) for p in chunk.path]

        if is_whisper:
            # Whisper always pads to 30 s; pooling must use the true frame count.
            feats = fe(waves, sampling_rate=SR, return_tensors="pt")
            inp = feats.input_features.to(device, dtype)
            out = model(inp, output_hidden_states=True)
            # 30 s -> 1500 frames, so 50 frames per second of real audio
            nfr = [max(1, min(1500, int(len(w) / SR * 50))) for w in waves]
            mask = torch.zeros(len(waves), 1500, device=device)
            for j, n in enumerate(nfr):
                mask[j, :n] = 1
        else:
            feats = fe(waves, sampling_rate=SR, return_tensors="pt", padding=True,
                       return_attention_mask=True)
            key = "input_features" if "input_features" in feats else "input_values"
            inp = feats[key].to(device, dtype)
            am = feats.get("attention_mask")
            am = am.to(device) if am is not None else None
            out = model(inp, attention_mask=am, output_hidden_states=True)
            nframes = out.hidden_states[0].shape[1]
            if am is not None and hasattr(model, "_get_feat_extract_output_lengths"):
                lens = model._get_feat_extract_output_lengths(am.sum(-1)).clamp(max=nframes)
                mask = (torch.arange(nframes, device=device)[None, :] < lens[:, None]).float()
            else:
                mask = torch.ones(len(waves), nframes, device=device)

        hs = torch.stack(out.hidden_states)                    # [L, B, T, D]
        m = mask[None, :, :, None].to(hs.dtype)
        pooled = (hs * m).sum(2) / m.sum(2).clamp(min=1)       # [L, B, D]
        pooled = pooled.float().cpu().numpy()
        nbad = int((~np.isfinite(pooled)).sum())
        if nbad:
            print(f"  WARNING: {nbad} non-finite values in this batch", flush=True)
        embs.append(pooled.astype(np.float16))
        utts.extend(chunk.utt_id.tolist())

        if i % (batch * 20) == 0:
            done = i + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{len(sub):,}  {rate:.1f} utt/s  "
                  f"eta {(len(sub)-done)/max(rate,1e-9)/60:.1f}m", flush=True)

    emb = np.concatenate(embs, axis=1)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{tag}.npz")
    np.savez_compressed(p, emb=emb, utt=np.array(utts))
    print(f"[{tag}] {emb.shape} (layers, utts, dim) -> {p}  "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--subset", default=os.path.join(ROOT, "eval/subset.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/emb"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="compute dtype; fp16 overflows in the deep layers of the "
                         "wav2vec2 models, bf16 has fp32 range at the same speed")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sub = pd.read_csv(a.subset, sep="\t", quoting=3, dtype={"speaker_id": str})
    if a.limit:
        sub = sub.head(a.limit)
    print(f"subset: {len(sub):,} utts")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    if "cuda" not in a.device:
        dtype = torch.float32
    print(f"compute dtype: {dtype}")
    for tag in a.models:
        repo = MODELS.get(tag, tag)
        print(f"\n=== {tag}  ({repo})", flush=True)
        try:
            run(tag, repo, sub, a.out, a.batch, a.max_s, a.device, dtype)
        except Exception as e:
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
