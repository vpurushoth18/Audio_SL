"""Sinhala speaker identification as a classification task.

Where the clustering tasks ask whether speakers fall out of the embedding space
on their own, this asks whether a simple probe can read speaker identity off the
embeddings. MTEB fits a logistic-regression classifier on a train split and
scores accuracy on a test split, so the datasets need both splits - see
scripts/prepare/prepare_openslr_speaker.py --splits.
"""

import os
from typing import Any

import mteb
from mteb.abstasks import AbsTaskClassification


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


class OpenSLR52SpeakerClassification(_LocalDatasetMixin, AbsTaskClassification):
    local_dir_name = "openslr52_speaker_clf"

    metadata = mteb.TaskMetadata(
        name="OpenSLR52SpeakerClassification",
        description=(
            "Sinhala speaker identification. Given a read-speech utterance from "
            "the OpenSLR-52 Sinhala ASR corpus, predict which of the 478 "
            "speakers produced it."
        ),
        reference="https://openslr.org/52/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR52-Speaker",
            "revision": "main",
        },
        type="AudioClassification",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2018-06-28", "2018-06-28"),
        domains=["Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
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

    input_column_name = "audio"
    label_column_name = "speaker_id"


class OpenSLR30SpeakerClassification(_LocalDatasetMixin, AbsTaskClassification):
    local_dir_name = "openslr30_speaker_clf"

    metadata = mteb.TaskMetadata(
        name="OpenSLR30SpeakerClassification",
        description=(
            "Sinhala speaker identification on the OpenSLR-30 crowd-sourced "
            "corpus: twelve speakers recorded at 48 kHz."
        ),
        reference="https://openslr.org/30/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR30-Speaker",
            "revision": "main",
        },
        type="AudioClassification",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="accuracy",
        date=("2017-10-03", "2017-10-03"),
        domains=["Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
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

    input_column_name = "audio"
    label_column_name = "speaker_id"
