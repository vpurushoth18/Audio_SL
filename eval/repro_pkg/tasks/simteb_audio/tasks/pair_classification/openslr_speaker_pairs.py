"""Sinhala same-speaker verification as a pair-classification task.

Given two utterances, decide whether they come from the same speaker. This is
the standard speaker-verification framing, and it tests something the clustering
and classification tasks do not: whether the distance between two embeddings is
directly meaningful, with no clusterer or probe in between.

MTEB scores it with max_ap - the best average precision over similarity
thresholds - the same metric SiMTEB's headline-prediction task uses.
"""

import os
from typing import Any

import mteb
from mteb.abstasks import AbsTaskPairClassification


class _LocalDatasetMixin:
    """Load from disk when SIMTEB_AUDIO_LOCAL is set, otherwise from the Hub."""

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


class OpenSLR52SpeakerPairClassification(
    _LocalDatasetMixin, AbsTaskPairClassification
):
    local_dir_name = "openslr52_speaker_pairs"

    metadata = mteb.TaskMetadata(
        name="OpenSLR52SpeakerPairClassification",
        description=(
            "Sinhala speaker verification. Given a pair of utterances from the "
            "OpenSLR-52 Sinhala ASR corpus, decide whether both were spoken by "
            "the same speaker. Positive and negative pairs are balanced."
        ),
        reference="https://openslr.org/52/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR52-SpeakerPairs",
            "revision": "main",
        },
        type="AudioPairClassification",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="max_ap",
        date=("2018-06-28", "2018-06-28"),
        domains=["Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{kjartansson-etal-sltu2018,
  address = {Gurugram, India},
  author = {Kjartansson, Oddur and Sarin, Supheakmungkol and Pipatsrisawat, Knot and Jansche, Martin and Ha, Linne},
  booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  month = {August},
  pages = {52--55},
  title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali}},
  year = {2018},
}
""",
    )

    input1_column_name = "audio1"
    input2_column_name = "audio2"
    label_column_name = "labels"


class OpenSLR30SpeakerPairClassification(
    _LocalDatasetMixin, AbsTaskPairClassification
):
    local_dir_name = "openslr30_speaker_pairs"

    metadata = mteb.TaskMetadata(
        name="OpenSLR30SpeakerPairClassification",
        description=(
            "Sinhala speaker verification on the OpenSLR-30 crowd-sourced "
            "corpus: twelve speakers recorded at 48 kHz."
        ),
        reference="https://openslr.org/30/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR30-SpeakerPairs",
            "revision": "main",
        },
        type="AudioPairClassification",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="max_ap",
        date=("2017-10-03", "2017-10-03"),
        domains=["Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{sodimana-etal-2018-step,
  address = {Gurugram, India},
  author = {Sodimana, Keshan and Pipatsrisawat, Knot and Ha, Linne and Jansche, Martin and Kjartansson, Oddur and De Silva, Pasindu and Sarin, Supheakmungkol},
  booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  month = {August},
  pages = {66--70},
  title = {{A Step-by-Step Process for Building TTS Voices Using Open Source Data and Frameworks for Bangla, Javanese, Khmer, Nepali, Sinhala, and Sundanese}},
  year = {2018},
}
""",
    )

    input1_column_name = "audio1"
    input2_column_name = "audio2"
    label_column_name = "labels"
