"""Sinhala tasks on WorldSpeech - found speech, as a contrast to read speech.

OpenSLR-52 and OpenSLR-30 are both prompted read speech recorded in controlled
conditions. WorldSpeech `si_lk` is found audio: ~21k segments taken from real
recordings, with a `source` field naming the recording each segment came from,
human and ASR transcripts, and per-segment quality measures (SNR, DNSMOS, CER).

That gives two things the OpenSLR tasks cannot:

  * a domain contrast - unscripted parliamentary speech versus read prompts
  * session clustering - group segments by sitting day. There are no speaker
    labels, but one session shares an acoustic environment and a pool of
    speakers, so this is a proxy for session/speaker structure.

Note on scope: every Sinhala segment in WorldSpeech comes from a single source,
`sri_lanka_parliament`, so this is formal unscripted speech from one venue
rather than varied web audio. `source` and `source_url` have one value each and
cannot define clusters; `session_date` is the usable grouping.

Set SIMTEB_AUDIO_LOCAL to run against the prepared local copy.
"""

import os
from typing import Any

import mteb
from mteb.abstasks import AbsTaskClassification, AbsTaskClustering

_CITATION = r"""
@misc{worldspeech,
  author = {{DISCO ETH}},
  title = {{WorldSpeech: a multilingual corpus of found speech}},
  howpublished = {\url{https://huggingface.co/datasets/disco-eth/WorldSpeech}},
}
"""


class _LocalDatasetMixin:
    local_dir_name: str

    def load_data(self, **kwargs: Any) -> None:
        root = os.environ.get("SIMTEB_AUDIO_LOCAL")
        if not root:
            super().load_data(**kwargs)
            return

        from datasets import load_from_disk

        self.dataset = load_from_disk(os.path.join(root, self.local_dir_name))
        self.dataset_transform()
        self.data_loaded = True


def _meta(name, description, task_type, main_score):
    return mteb.TaskMetadata(
        name=name,
        description=description,
        reference="https://huggingface.co/datasets/disco-eth/WorldSpeech",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-WorldSpeech-Sinhala",
            "revision": "main",
        },
        type=task_type,
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score=main_score,
        date=("2024-01-01", "2025-12-31"),
        domains=["Spoken", "Web"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION,
    )


class WorldSpeechSinhalaSessionClustering(_LocalDatasetMixin, AbsTaskClustering):
    local_dir_name = "worldspeech_sin_session"

    metadata = _meta(
        "WorldSpeechSinhalaSessionClustering",
        "Cluster Sinhala parliamentary speech segments by recording session "
        "(sitting day). Unscripted speech rather than read prompts, so this "
        "measures session and environment structure outside the studio.",
        "AudioClustering",
        "v_measure",
    )

    input_column_name = "audio"
    label_column_name = "session_id"


class WorldSpeechSinhalaQualityClassification(
    _LocalDatasetMixin, AbsTaskClassification
):
    """Recording quality is a label the corpus gives for free.

    Segments are bucketed by DNSMOS overall score into low / medium / high. A
    model whose embeddings encode acoustic quality separates these; one that
    encodes only linguistic content does not. Useful as a diagnostic: strong
    quality separation on an otherwise weak model suggests the embedding is
    dominated by channel rather than speech.
    """

    local_dir_name = "worldspeech_sin_quality"

    metadata = _meta(
        "WorldSpeechSinhalaQualityClassification",
        "Classify Sinhala found-speech segments into recording-quality bands "
        "derived from DNSMOS overall scores. A diagnostic for how much acoustic "
        "channel information the embedding carries.",
        "AudioClassification",
        "accuracy",
    )

    input_column_name = "audio"
    label_column_name = "quality_band"
