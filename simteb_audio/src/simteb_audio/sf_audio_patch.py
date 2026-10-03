"""Decode HF `Audio` columns with soundfile instead of torchcodec.

`datasets` 4.x routes every Audio column through torchcodec, which cannot be
installed on this cluster: the current wheel targets CUDA 13 (our torch is
cu124) and the older wheels need FFmpeg shared libraries that are not present,
with `conda install ffmpeg` refused because it would remove conda's own tqdm.

Nothing about decoding actually requires torchcodec here. libsndfile reads
WAV/FLAC/OGG/Opus/MP3, so this replaces `Audio.decode_example` with a soundfile
implementation returning the pre-4.x dict shape:

    {"path": str | None, "array": np.float32[n], "sampling_rate": int}

which is exactly what MTEB's AudioCollator reads (`row["audio"]["array"]`).

Import this before any task loads data:

    import simteb_audio.sf_audio_patch  # noqa: F401

It is a no-op if torchcodec is importable, so it disappears on a normal machine.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _available() -> bool:
    try:
        import torchcodec  # noqa: F401

        return True
    except Exception:
        return False


def apply() -> bool:
    """Patch datasets.Audio.decode_example. Returns True if the patch was applied."""
    global _PATCHED
    if _PATCHED or _available():
        return False

    import numpy as np
    import soundfile as sf
    from datasets.features.audio import Audio

    def decode_example(self, value, token_per_repo_id=None):  # noqa: ANN001
        if value is None:
            return None

        # {"path": ..., "bytes": ...} is what Arrow stores
        path, raw = None, None
        if isinstance(value, dict):
            path, raw = value.get("path"), value.get("bytes")
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        elif isinstance(value, str):
            path = value
        else:
            # already decoded, or a type we do not handle - hand it back
            return value

        if raw is not None:
            src = io.BytesIO(raw)
        elif path is not None:
            src = path
        else:
            raise ValueError("Audio value has neither bytes nor path")

        array, sampling_rate = sf.read(src, dtype="float32", always_2d=False)
        if array.ndim > 1:                      # mixdown; mono is what encoders want
            array = array.mean(axis=1)

        target = getattr(self, "sampling_rate", None)
        if target and target != sampling_rate:
            import soxr

            array = soxr.resample(array, sampling_rate, target)
            sampling_rate = target

        return {"path": path, "array": np.asarray(array, dtype=np.float32),
                "sampling_rate": int(sampling_rate)}

    Audio.decode_example = decode_example
    _PATCHED = True
    logger.info("patched datasets Audio.decode_example to use soundfile")
    return True


_APPLIED = apply()
