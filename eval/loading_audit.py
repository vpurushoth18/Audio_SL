#!/usr/bin/env python3
"""Diagnose model loading for the suspect checkpoints. Read-only.

Writes nothing into eval/analysis or either result cache. Its own report goes
where the caller asks.

Three tests per checkpoint:

  1. LOAD REPORT. Instantiate through the registry's own loader, exactly as the
     run did, and capture transformers' missing / unexpected key report. The
     wrapper path is what matters: testing `Wav2Vec2Model.from_pretrained` by
     hand can dispatch to a different class than the wrapper gets, which is how
     this defect was missed the first time.

  2. DOUBLE-LOAD DETERMINISM. Load the backbone twice and compare a weight
     tensor. Weights restored from a checkpoint are identical across loads;
     weights that were randomly initialised differ. This is decisive and needs
     no reference values.

  3. EMBEDDING VARIANCE. Encode a handful of real utterances and measure the
     spread across them. A backbone that did not load tends to collapse to a
     near-constant vector, which is what produces chance-level scores.

For the WavLM variants the question is different: the scores are exact
duplicates, so the test is whether the fine-tuned checkpoints carry different
backbone weights at all, compared tensor by tensor.
"""
import argparse
import io
import json
import os
import sys
import contextlib

import numpy as np
import torch

ROOT = "/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL"
sys.path.insert(0, os.path.join(ROOT, "simteb_audio/src"))

SUSPECT = ["facebook/wav2vec2-conformer-rel-pos-large",
           "facebook/wav2vec2-xls-r-300m-21-to-en"]
# a known-good control: same wrapper, same family, scores look normal
CONTROL = ["facebook/wav2vec2-large-robust"]
WAVLM = ["microsoft/wavlm-base", "microsoft/wavlm-base-sv",
         "microsoft/wavlm-base-plus", "microsoft/wavlm-base-plus-sv",
         "microsoft/wavlm-base-plus-sd"]


def wrapper_of(name):
    import mteb
    meta = mteb.get_model_meta(name)
    ld = meta.loader
    cls = ld.func if hasattr(ld, "func") else ld
    return meta, cls


def capture_load(name):
    """Instantiate via the registry loader, capturing the transformers report."""
    import mteb
    buf_out, buf_err = io.StringIO(), io.StringIO()
    err = None
    model = None
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            model = mteb.get_model(name, device="cpu")
        except Exception as e:                      # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    text = buf_out.getvalue() + buf_err.getvalue()
    return model, text, err


def parse_report(text):
    miss = [l for l in text.split("\n") if "MISSING" in l and "|" in l]
    unex = [l for l in text.split("\n") if "UNEXPECTED" in l and "|" in l]
    return miss, unex


def backbone_of(model):
    """The nn.Module that actually produces the representations."""
    for attr in ("model", "encoder", "backbone"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    return model if isinstance(model, torch.nn.Module) else None


def first_weight(mod, keys=("feature_extractor", "encoder.layers.0")):
    sd = mod.state_dict()
    for k in sd:
        if any(kk in k for kk in keys) and sd[k].dtype.is_floating_point:
            return k, sd[k].detach().float().clone()
    k = next(iter(sd))
    return k, sd[k].detach().float().clone()


def double_load(name):
    """Random-initialised weights differ between loads; restored ones do not."""
    import transformers
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    out = {}
    tensors = []
    for _ in range(2):
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            from transformers import Wav2Vec2Model
            try:
                m = Wav2Vec2Model.from_pretrained(name)
            except Exception as e:                  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}
        k, t = first_weight(m)
        tensors.append((k, t))
        del m
    k0, t0 = tensors[0]
    k1, t1 = tensors[1]
    same = bool(torch.equal(t0, t1))
    out.update(tested_key=k0, identical_across_loads=same,
               max_abs_diff=float((t0 - t1).abs().max()),
               verdict=("weights restored from checkpoint" if same
                        else "RANDOMLY INITIALISED - differs between loads"),
               model_class_via_wav2vec2model=type(m).__name__
               if 'm' in dir() else None,
               config_model_type=cfg.model_type)
    return out


def embed_sample(model, n=8, seed=0):
    """Encode a few real utterances; report spread across them."""
    import pandas as pd
    import soundfile as sf
    import soxr
    from torch.utils.data import DataLoader

    sub = pd.read_csv(os.path.join(ROOT, "eval/subset.tsv"), sep="\t", quoting=3)
    rng = np.random.default_rng(seed)
    pick = sub.iloc[rng.choice(len(sub), n, replace=False)]
    items = []
    for p in pick.path:
        x, sr = sf.read(os.path.join(ROOT, p), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            x = soxr.resample(x, sr, 16000)
        items.append({"audio": {"array": x, "sampling_rate": 16000}})

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i): return items[i]

    dl = DataLoader(DS(), batch_size=4, collate_fn=lambda b: {
        "audio": [x["audio"] for x in b]})
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        E = model.get_audio_embeddings(dl, show_progress_bar=False)
    E = np.asarray(E, dtype=np.float64)
    # pairwise cosine between distinct utterances
    Z = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    C = Z @ Z.T
    off = C[~np.eye(len(C), dtype=bool)]
    return dict(shape=list(E.shape),
                per_dim_std_mean=float(E.std(axis=0).mean()),
                across_utterance_std=float(E.std(axis=0).mean()),
                mean_pairwise_cosine=float(off.mean()),
                min_pairwise_cosine=float(off.min()),
                n_distinct_rows=int(len({tuple(np.round(r, 6)) for r in E})),
                degenerate=bool(off.mean() > 0.999 or E.std(axis=0).mean() < 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/loading_audit")
    ap.add_argument("--skip-embed", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import simteb_audio.models.expansion_models  # noqa: F401  registers

    report = {}

    # ---------------- part 1: suspects + control
    for name in SUSPECT + CONTROL:
        print(f"\n{'='*72}\n{name}\n{'='*72}", flush=True)
        rec = {}
        meta, cls = wrapper_of(name)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(name)
        rec["wrapper_class"] = cls.__name__
        rec["revision_declared"] = meta.revision
        rec["config_model_type"] = cfg.model_type
        rec["config_hidden_size"] = getattr(cfg, "hidden_size", None)
        rec["config_architectures"] = getattr(cfg, "architectures", None)
        print(f"  wrapper={cls.__name__}  config.model_type={cfg.model_type}  "
              f"architectures={rec['config_architectures']}")

        model, text, err = capture_load(name)
        miss, unex = parse_report(text)
        rec["load_error"] = err
        rec["n_missing_key_lines"] = len(miss)
        rec["n_unexpected_key_lines"] = len(unex)
        rec["missing_keys_sample"] = [l.split("|")[0].strip() for l in miss[:25]]
        rec["unexpected_keys_sample"] = [l.split("|")[0].strip() for l in unex[:15]]
        if model is not None:
            bb = backbone_of(model)
            rec["instantiated_model_class"] = type(getattr(model, "model", model)).__name__
            rec["backbone_class"] = type(bb).__name__ if bb is not None else None
            rec["backbone_n_params_m"] = (
                round(sum(p.numel() for p in bb.parameters()) / 1e6, 1)
                if bb is not None else None)
        print(f"  instantiated={rec.get('instantiated_model_class')}  "
              f"missing_lines={len(miss)}  unexpected_lines={len(unex)}")
        if miss:
            print("  MISSING (first 6):")
            for l in rec["missing_keys_sample"][:6]:
                print(f"     {l}")

        rec["double_load"] = double_load(name)
        print(f"  double-load: {rec['double_load'].get('verdict')}")

        if model is not None and not a.skip_embed:
            try:
                rec["embedding_check"] = embed_sample(model)
                ec = rec["embedding_check"]
                print(f"  embeddings: shape={ec['shape']} "
                      f"mean_pairwise_cos={ec['mean_pairwise_cosine']:.4f} "
                      f"across_utt_std={ec['across_utterance_std']:.6f} "
                      f"degenerate={ec['degenerate']}")
            except Exception as e:                  # noqa: BLE001
                rec["embedding_check"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  embeddings: FAILED {rec['embedding_check']['error']}")
        del model
        report[name] = rec

    # ---------------- part 2: wavlm backbone comparison
    print(f"\n{'='*72}\nWavLM variants: backbone weight comparison\n{'='*72}",
          flush=True)
    from transformers import WavLMModel
    sds = {}
    for name in WAVLM:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            m = WavLMModel.from_pretrained(name)
        sds[name] = {k: v.detach().float().clone()
                     for k, v in m.state_dict().items()}
        del m
        print(f"  loaded {name}: {len(sds[name])} tensors")

    wl = {}
    base = "microsoft/wavlm-base"
    for pair in [("microsoft/wavlm-base", "microsoft/wavlm-base-sv"),
                 ("microsoft/wavlm-base-plus", "microsoft/wavlm-base-plus-sv"),
                 ("microsoft/wavlm-base-plus", "microsoft/wavlm-base-plus-sd"),
                 ("microsoft/wavlm-base", "microsoft/wavlm-base-plus")]:
        A, B = sds[pair[0]], sds[pair[1]]
        shared = [k for k in A if k in B and A[k].shape == B[k].shape]
        diffs = {k: float((A[k] - B[k]).abs().max()) for k in shared}
        nz = [k for k, v in diffs.items() if v > 0]
        wl[f"{pair[0]} vs {pair[1]}"] = dict(
            n_shared_tensors=len(shared),
            n_tensors_differing=len(nz),
            max_abs_diff=max(diffs.values()) if diffs else None,
            identical_backbone=len(nz) == 0,
            examples_differing=sorted(nz, key=lambda k: -diffs[k])[:5])
        print(f"  {pair[0].split('/')[1]:20s} vs {pair[1].split('/')[1]:20s} "
              f"shared={len(shared):3d} differing={len(nz):3d} "
              f"maxdiff={max(diffs.values()) if diffs else float('nan'):.6g} "
              f"-> {'IDENTICAL' if not nz else 'different'}")
    report["wavlm_backbone_comparison"] = wl

    json.dump(report, open(os.path.join(a.out, "loading_audit.json"), "w"),
              indent=2, default=str)
    print(f"\n  wrote {os.path.join(a.out, 'loading_audit.json')}")


if __name__ == "__main__":
    main()
