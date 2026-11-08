"""Registry entries for the checkpoints added in the model expansion.

MTEB ships wrappers for every family here - Wav2Vec2AudioWrapper, HubertWrapper,
WavlmWrapper, WhisperAudioWrapper, UniSpeechWrapper, Data2VecAudioWrapper - but
registers only a handful of checkpoints per family. The expansion needs sizes
and fine-tuned variants that are not in the default registry, so each one gets a
ModelMeta pointing at the family's existing wrapper.

Nothing here changes how a model is encoded. The extraction rule is the wrapper's
and is identical to the one used for the original twelve checkpoints, which is
the condition for the cross-benchmark correlation to mean anything.

Revisions are resolved from the Hub at registration time and recorded in
eval/analysis/expansion_resolved_revisions.tsv by the runner, because `main` is
not an immutable reference.
"""
from __future__ import annotations

from mteb.models import ModelMeta
from mteb.models.model_implementations import MODEL_REGISTRY
from mteb.models.model_implementations.data2vec_models import Data2VecAudioWrapper
from mteb.models.model_implementations.hubert_models import HubertWrapper
from mteb.models.model_implementations.unispeech_models import UniSpeechWrapper
from mteb.models.model_implementations.wav2vec2_models import Wav2Vec2AudioWrapper
from mteb.models.model_implementations.wavlm_models import WavlmWrapper
from mteb.models.model_implementations.whisper_models import WhisperAudioWrapper

# name, wrapper, encoder params, embed dim, licence, release date
SPECS = [
    # --- wav2vec2 family
    ("facebook/wav2vec2-base", Wav2Vec2AudioWrapper, 94_000_000, 768,
     "apache-2.0", "2020-06-20"),
    ("facebook/wav2vec2-base-960h", Wav2Vec2AudioWrapper, 94_000_000, 768,
     "apache-2.0", "2020-06-20"),
    ("facebook/wav2vec2-large-960h", Wav2Vec2AudioWrapper, 315_000_000, 1024,
     "apache-2.0", "2020-06-20"),
    ("facebook/wav2vec2-large-robust", Wav2Vec2AudioWrapper, 315_000_000, 1024,
     "apache-2.0", "2021-04-20"),
    ("facebook/wav2vec2-large-xlsr-53", Wav2Vec2AudioWrapper, 315_000_000, 1024,
     "apache-2.0", "2020-12-11"),
    ("facebook/wav2vec2-xls-r-300m-21-to-en", Wav2Vec2AudioWrapper, 315_000_000,
     1024, "apache-2.0", "2021-11-12"),
    # --- hubert
    ("facebook/hubert-base-ls960", HubertWrapper, 94_000_000, 768,
     "apache-2.0", "2021-06-14"),
    ("facebook/hubert-large-ll60k", HubertWrapper, 316_000_000, 1024,
     "apache-2.0", "2021-06-14"),
    ("facebook/hubert-xlarge-ll60k", HubertWrapper, 964_000_000, 1280,
     "apache-2.0", "2021-06-14"),
    # --- unispeech-sat
    ("microsoft/unispeech-sat-base", UniSpeechWrapper, 94_000_000, 768,
     "mit", "2021-10-12"),
    ("microsoft/unispeech-sat-base-plus", UniSpeechWrapper, 94_000_000, 768,
     "mit", "2021-10-12"),
    ("microsoft/unispeech-sat-large", UniSpeechWrapper, 316_000_000, 1024,
     "mit", "2021-10-12"),
    # --- data2vec
    ("facebook/data2vec-audio-base", Data2VecAudioWrapper, 93_000_000, 768,
     "apache-2.0", "2022-03-07"),
    ("facebook/data2vec-audio-base-960h", Data2VecAudioWrapper, 93_000_000, 768,
     "apache-2.0", "2022-03-07"),
    ("facebook/data2vec-audio-large", Data2VecAudioWrapper, 313_000_000, 1024,
     "apache-2.0", "2022-03-07"),
    # --- whisper, encoder only
    ("openai/whisper-tiny.en", WhisperAudioWrapper, 8_000_000, 384,
     "apache-2.0", "2022-09-27"),
    ("openai/whisper-base.en", WhisperAudioWrapper, 21_000_000, 512,
     "apache-2.0", "2022-09-27"),
    ("openai/whisper-small.en", WhisperAudioWrapper, 88_000_000, 768,
     "apache-2.0", "2022-09-27"),
    ("openai/whisper-medium.en", WhisperAudioWrapper, 307_000_000, 1024,
     "apache-2.0", "2022-09-27"),
    ("openai/whisper-large-v2", WhisperAudioWrapper, 637_000_000, 1280,
     "apache-2.0", "2022-12-05"),
    # --- wavlm speaker-verification variant
    ("microsoft/wavlm-base-plus-sv", WavlmWrapper, 94_000_000, 768,
     "mit", "2021-10-26"),
    # --- wav2vec2-conformer: NEW ARCHITECTURE FAMILY.
    # Substituted for speechbrain/m-ctc-t-large, whose MTEB wrapper pins
    # transformers<5 (M-CTC-T was removed in v5) and so cannot run in the
    # environment the other 44 checkpoints were evaluated in.
    # Wav2Vec2Model.from_pretrained dispatches on the config to
    # Wav2Vec2ConformerModel; the only discarded keys are the pretraining
    # heads (quantizer, project_q, project_hid), so the encoder loads whole.
    ("facebook/wav2vec2-conformer-rel-pos-large", Wav2Vec2AudioWrapper,
     618_000_000, 1024, "apache-2.0", "2022-05-06"),
]


def register() -> list[str]:
    """Add every expansion checkpoint MTEB does not already know. Idempotent."""
    added = []
    for name, wrapper, params, dim, lic, date in SPECS:
        if name in MODEL_REGISTRY:
            continue
        MODEL_REGISTRY[name] = ModelMeta(
            loader=wrapper,
            name=name,
            languages=["eng-Latn"],
            open_weights=True,
            revision="main",
            release_date=date,
            max_tokens=None,
            n_parameters=params,
            memory_usage_mb=int(params / 1e6 * 4),
            embed_dim=dim,
            license=lic,
            reference=f"https://huggingface.co/{name}",
            similarity_fn_name="cosine",
            framework=["PyTorch"],
            use_instructions=False,
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
            modalities=["audio"],
        )
        added.append(name)
    return added


register()
