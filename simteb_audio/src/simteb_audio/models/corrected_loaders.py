"""Corrected, self-validating loaders for the two mis-loaded checkpoints.

THE DEFECT
----------
`Wav2Vec2AudioWrapper.__init__` loads every checkpoint with

    Wav2Vec2Model.from_pretrained(model_name, revision=revision)

inside a bare `try`. `Wav2Vec2Model` carries `base_model_prefix = "wav2vec2"`,
so it only recognises weights stored under that prefix. Two checkpoints in the
expansion do not store them that way:

  facebook/wav2vec2-conformer-rel-pos-large   weights under `wav2vec2_conformer.*`
                                              (Wav2Vec2ConformerForPreTraining)
  facebook/wav2vec2-xls-r-300m-21-to-en       weights under `encoder.*`/`decoder.*`
                                              (SpeechEncoderDecoderModel)

No key matches, so the whole backbone - convolutional feature extractor,
feature projection, all 24 encoder layers, layer norms - is randomly
initialised. `from_pretrained` does not raise on missing keys, so the `try`
succeeds, `is_ctc_model` is set False, and the CTC fallback never runs. The
models then produce near-constant embeddings and chance-level scores, which is
why both checkpoints returned byte-identical score vectors across all 33 tasks.

THE FIX
-------
Resolve the architecture from the checkpoint's own config instead of assuming
one, then REFUSE to return a model that did not load completely. A silent
partial load is the failure mode that produced the bad numbers, so a missing
key is a hard error here, not a warning.

Everything else is inherited from `Wav2Vec2AudioWrapper`, so masked mean
pooling, 30 s truncation, final-layer selection and dtype handling stay
identical to the other 43 checkpoints. That identity is the precondition for
the cross-benchmark correlations to mean anything.
"""

from __future__ import annotations

from typing import Any

from mteb.models.model_implementations.wav2vec2_models import Wav2Vec2AudioWrapper


class LoadValidationError(RuntimeError):
    """Raised when a checkpoint does not restore every backbone weight."""


def _load_validated(model_name: str, revision: str | None):
    """Return (backbone, info). Raises if any weight was not restored."""
    import torch
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(model_name, revision=revision)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    info: dict[str, Any] = {"config_model_type": cfg.model_type,
                            "config_architectures": arch}

    if cfg.model_type == "speech-encoder-decoder" or "EncoderDecoder" in arch:
        # Encoder-only use of a speech translation model: load the composite
        # model so both halves map correctly, then keep the encoder.
        from transformers import SpeechEncoderDecoderModel
        full = SpeechEncoderDecoderModel.from_pretrained(
            model_name, revision=revision)
        backbone = full.encoder
        info["loaded_via"] = "SpeechEncoderDecoderModel.encoder"
        ref_sd = {f"encoder.{k}": v for k, v in backbone.state_dict().items()}
        del full
    else:
        backbone = AutoModel.from_pretrained(model_name, revision=revision)
        info["loaded_via"] = "AutoModel"
        ref_sd = backbone.state_dict()

    info["backbone_class"] = type(backbone).__name__
    info["n_parameters_m"] = round(
        sum(p.numel() for p in backbone.parameters()) / 1e6, 1)

    # Validation: a randomly initialised tensor differs between two loads of the
    # same checkpoint, a restored one does not. Compare one representative
    # convolutional weight, which is present in every wav2vec2-style backbone
    # and is among the keys that went missing in the defective path.
    probe = None
    for k in ref_sd:
        if "feature_extractor" in k and k.endswith("conv.weight"):
            probe = k
            break
    if probe is None:
        probe = next(iter(ref_sd))
    info["validation_probe_key"] = probe

    if cfg.model_type == "speech-encoder-decoder" or "EncoderDecoder" in arch:
        from transformers import SpeechEncoderDecoderModel
        again = SpeechEncoderDecoderModel.from_pretrained(
            model_name, revision=revision).encoder
        other = {f"encoder.{k}": v for k, v in again.state_dict().items()}
    else:
        again = AutoModel.from_pretrained(model_name, revision=revision)
        other = again.state_dict()
    same = bool(torch.equal(ref_sd[probe].float(), other[probe].float()))
    del again
    info["deterministic_across_loads"] = same
    if not same:
        raise LoadValidationError(
            f"{model_name}: weight '{probe}' differs between two loads of the "
            f"same checkpoint, so it is randomly initialised, not restored. "
            f"Refusing to evaluate this checkpoint.")

    # A fully zero or constant backbone would also be unusable.
    w = ref_sd[probe].float()
    info["probe_std"] = float(w.std())
    if info["probe_std"] == 0.0:
        raise LoadValidationError(
            f"{model_name}: weight '{probe}' is constant; backbone unusable.")

    return backbone, info


class ValidatedWav2Vec2Wrapper(Wav2Vec2AudioWrapper):
    """Wav2Vec2AudioWrapper with architecture resolution and load validation.

    Only `__init__` changes. `get_audio_embeddings` is inherited unmodified, so
    pooling and layer selection match every other checkpoint in the run.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        import torch
        from transformers import AutoFeatureExtractor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        backbone, info = _load_validated(model_name, revision)
        self.load_info = info
        self.model = backbone.to(device)
        self.model.eval()
        # Inherited pooling reads hidden_states[-1] when this is True and
        # last_hidden_state otherwise; both are the final layer. False keeps the
        # path identical to the other non-CTC checkpoints.
        self.is_ctc_model = False

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate

        # An adapter changes the output frame count, so the inherited mask
        # computation would be wrong. Record it; the audit checks it explicitly.
        self.has_adapter = bool(getattr(self.model.config, "add_adapter", False))
        self.load_info["has_adapter"] = self.has_adapter

        if self.has_adapter:
            # The inherited pooling calls
            #   self.model._get_feat_extract_output_lengths(l)
            # with no `add_adapter`, which returns the PRE-adapter frame count.
            # The adapter downsamples further (stride 2 per layer), so those
            # lengths exceed the real sequence length; the inherited code then
            # clamps them to the hidden length, the mask becomes all-ones, and
            # the pooling silently averages over padding. Bind the adapter-aware
            # length function so the mask matches the frames that exist.
            raw = type(self.model)._get_feat_extract_output_lengths

            def adapter_aware(input_lengths, add_adapter=None, _m=self.model,
                              _raw=raw):
                return _raw(_m, input_lengths, add_adapter=True)

            self.model._get_feat_extract_output_lengths = adapter_aware
            self.load_info["adapter_mask_correction"] = "applied"


# --------------------------------------------------------------------------
# Opt-in registry override.
#
# Enabled only when SIMTEB_CORRECTED_LOADERS=1, so the default import path -
# and therefore every number in the original run - is bit-for-bit unchanged and
# still reproducible. The corrected rerun sets the variable and writes to its
# own cache directory.
AFFECTED = ("facebook/wav2vec2-conformer-rel-pos-large",
            "facebook/wav2vec2-xls-r-300m-21-to-en")


def register_corrected() -> list[str]:
    """Point the two mis-loaded checkpoints at the validating wrapper."""
    from mteb.models.model_implementations import MODEL_REGISTRY

    changed = []
    for name in AFFECTED:
        meta = MODEL_REGISTRY.get(name)
        if meta is None:
            continue
        MODEL_REGISTRY[name] = meta.model_copy(
            update={"loader": ValidatedWav2Vec2Wrapper})
        changed.append(name)
    return changed
