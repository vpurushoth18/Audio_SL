"""Pooled-MFCC baseline, registered as an MTEB model.

The audit's first pass compared a hand-rolled MFCC probe against MTEB-protocol
encoder scores. Those are different protocols, so the comparison was indicative
only: MTEB fits its classification probe with samples_per_label=8 over
n_experiments=10, uses its own clustering and pair-AP implementations, and
applies its own splits.

Registering the baseline as a model removes that objection entirely. It then
runs through exactly the same evaluator, splits and metric code as every deep
encoder, and appears in the results table as another row.

The feature is deliberately minimal and has no learned parameters:

    40 mel filters -> log -> DCT-II -> keep 20 coefficients
    -> concatenate mean and std over frames  => 40 dimensions

If this matches a deep encoder on a task, the task is solvable from pooled
spectral statistics and does not measure representation quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations import MODEL_REGISTRY
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.types import Array
    from mteb.types._encoder_io import AudioInput

SR = 16000
N_MELS = 40
N_MFCC = 20


def _mel_filterbank(n_bins: int, sr: int = SR, n_mels: int = N_MELS):
    f_hz = np.linspace(0.0, sr / 2, n_bins)

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(20.0), hz2mel(sr / 2), n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (f_hz >= lo) & (f_hz <= mid)
        right = (f_hz > mid) & (f_hz <= hi)
        fb[i, left] = (f_hz[left] - lo) / max(mid - lo, 1e-9)
        fb[i, right] = (hi - f_hz[right]) / max(hi - mid, 1e-9)
    return fb


class MFCCBaselineWrapper(AbsEncoder):
    """Pooled MFCC mean+std. No parameters, no training, CPU only."""

    def __init__(
        self,
        model_name: str = "baseline/mfcc-40",
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: float = 20.0,
        n_fft: int = 512,
        hop: int = 160,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.model_revision = revision
        self.device = "cpu"
        self.max_audio_length_seconds = max_audio_length_seconds
        self.n_fft = n_fft
        self.hop = hop
        self.sampling_rate = SR
        self._fb = _mel_filterbank(n_fft // 2 + 1)
        k = np.arange(N_MELS)
        self._dct = np.cos(
            np.pi / N_MELS * (k[None, :] + 0.5) * np.arange(N_MFCC)[:, None]
        ).astype(np.float32)

    def _features(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = x[: int(self.max_audio_length_seconds * SR)]
        if x.size < self.n_fft * 2:
            x = np.pad(x, (0, self.n_fft * 2 - x.size))
        n_frames = 1 + (len(x) - self.n_fft) // self.hop
        idx = (np.arange(self.n_fft)[None, :]
               + self.hop * np.arange(n_frames)[:, None])
        frames = x[idx] * np.hanning(self.n_fft).astype(np.float32)[None, :]
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        mel = np.log(power @ self._fb.T + 1e-10)
        c = mel @ self._dct.T
        return np.concatenate([c.mean(0), c.std(0)]).astype(np.float32)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(
            target_sampling_rate=self.sampling_rate,
            max_samples=int(self.max_audio_length_seconds * SR),
        )
        out = []
        for batch in tqdm(inputs, disable=not show_progress_bar):
            out.append(np.stack([self._features(a["array"])
                                 for a in batch["audio"]]))
        return np.concatenate(out, axis=0)

    def encode(self, inputs: DataLoader[AudioInput], **kwargs: Any) -> Array:
        if "audio" not in inputs.dataset.features:
            raise ValueError("MFCCBaselineWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


mfcc_baseline = ModelMeta(
    loader=MFCCBaselineWrapper,
    name="baseline/mfcc-40",
    languages=["sin-Sinh", "eng-Latn"],   # language-agnostic by construction
    open_weights=True,
    revision="1",
    release_date="2026-10-02",
    max_tokens=None,
    n_parameters=0,
    memory_usage_mb=0,
    embed_dim=2 * N_MFCC,
    license="mit",
    reference="https://github.com/vpurushoth18/Audio_SL",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


def register() -> list[str]:
    """Add the baseline to MTEB's registry. Idempotent."""
    if mfcc_baseline.name not in MODEL_REGISTRY:
        MODEL_REGISTRY[mfcc_baseline.name] = mfcc_baseline
        return [mfcc_baseline.name]
    return []


register()
