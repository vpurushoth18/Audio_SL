"""Model entries MTEB does not ship, registered into its own registry.

MTEB registers only the 1B MMS checkpoints and has no w2v-BERT support at all,
so `mteb.get_model("facebook/mms-300m")` falls through to the SentenceTransformer
loader and dies on a raw audio repo. Importing this module adds both models to
MTEB's MODEL_REGISTRY, after which they load like any built-in model.

  import simteb_audio.models.extra_audio_models  # noqa: F401
  model = mteb.get_model("facebook/w2v-bert-2.0")

mms-300m reuses MTEB's own MMSWrapper. w2v-BERT needs a new wrapper: it is a
Conformer taking pre-computed filterbank features rather than a raw waveform, so
Wav2Vec2FeatureExtractor/Wav2Vec2Model do not apply. The pooling below is the
same masked mean MTEB uses elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations import MODEL_REGISTRY
from mteb.models.model_implementations.mms_models import MMSWrapper
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.types import Array
    from mteb.types._encoder_io import AudioInput


class W2VBert2Wrapper(AbsEncoder):
    """Encoder for facebook/w2v-bert-2.0 (Wav2Vec2-BERT 2.0).

    Unlike wav2vec2, this model consumes log-mel filterbank features produced by
    a SeamlessM4T feature extractor, so the input key is `input_features` and the
    frame mask has to come from the model's own downsampling rule.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.model_revision = revision
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, revision=revision
        )
        self.model = Wav2Vec2BertModel.from_pretrained(
            model_name, revision=revision
        ).to(device)
        self.model.eval()
        self.sampling_rate = self.feature_extractor.sampling_rate

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        import numpy as np
        import torch

        inputs.collate_fn = AudioCollator(
            target_sampling_rate=self.sampling_rate,
            max_samples=int(self.max_audio_length_seconds * self.sampling_rate),
        )

        all_embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar):
            waveforms = [audio["array"] for audio in batch["audio"]]

            feats = self.feature_extractor(
                waveforms,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_features = feats["input_features"].to(self.device)
            attention_mask = feats.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_features, attention_mask=attention_mask
                ).last_hidden_state                      # [B, T, D]

            n_frames = out.shape[1]
            if attention_mask is not None:
                # the encoder downsamples, so the input mask cannot be used as-is;
                # ask the model how many output frames each input length gives
                lengths = self.model._get_feat_extract_output_lengths(
                    attention_mask.sum(dim=-1)
                ).clamp(max=n_frames)
                mask = (
                    torch.arange(n_frames, device=self.device)[None, :]
                    < lengths[:, None]
                ).to(out.dtype)
            else:
                mask = torch.ones(
                    out.shape[0], n_frames, device=self.device, dtype=out.dtype
                )

            mask = mask.unsqueeze(-1)                      # [B, T, 1]
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def encode(self, inputs: DataLoader[AudioInput], **kwargs: Any) -> Array:
        if "audio" not in inputs.dataset.features:
            raise ValueError("W2VBert2Wrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


mms_300m = ModelMeta(
    loader=MMSWrapper,
    name="facebook/mms-300m",
    languages=["sin-Sinh", "eng-Latn"],  # pretrained on 1400+ languages
    open_weights=True,
    revision="4ee317ce793c53dbc041fc4376c7558292dd38dc",
    release_date="2023-05-22",
    max_tokens=None,
    n_parameters=315_000_000,
    memory_usage_mb=1200,
    embed_dim=1024,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/facebook/mms-300m",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/fairseq/tree/main/examples/mms",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

w2v_bert_2_0 = ModelMeta(
    loader=W2VBert2Wrapper,
    name="facebook/w2v-bert-2.0",
    languages=["sin-Sinh", "eng-Latn"],  # pretrained on 143+ languages
    open_weights=True,
    revision="da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    release_date="2023-12-01",
    max_tokens=None,
    n_parameters=580_000_000,
    memory_usage_mb=2300,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/w2v-bert-2.0",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/seamless_communication",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


def register() -> list[str]:
    """Add the models to MTEB's registry. Idempotent."""
    added = []
    for meta in (mms_300m, w2v_bert_2_0):
        if meta.name not in MODEL_REGISTRY:
            MODEL_REGISTRY[meta.name] = meta
            added.append(meta.name)
    return added


register()
