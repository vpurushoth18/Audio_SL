"""Sinhala speaker clustering tasks, built on OpenSLR-52 and OpenSLR-30.

Audio counterparts to the SiMTEB text tasks: cluster utterances by speaker
without seeing the labels, then compare the partition against the true
speakers. MTEB scores this with V-measure.

Set SIMTEB_AUDIO_LOCAL to a directory saved by scripts/prepare to run against
a local dataset instead of the Hub - useful before the dataset is pushed.
"""

import os
from typing import Any

import mteb
from mteb.abstasks import AbsTaskClustering


class _LocalDatasetMixin:
    """Load from disk when SIMTEB_AUDIO_LOCAL is set, otherwise from the Hub.

    MTEB's own load_data goes straight to `datasets.load_dataset` with the
    metadata path and revision, which cannot be satisfied until the dataset has
    been pushed. This keeps the task runnable while the data is still local.
    """

    local_dir_name: str

    def load_data(self, **kwargs: Any) -> None:
        root = os.environ.get("SIMTEB_AUDIO_LOCAL")
        if not root:
            super().load_data(**kwargs)
            return

        from datasets import load_from_disk

        path = os.path.join(root, self.local_dir_name)
        self.dataset = load_from_disk(path)
        self.dataset_transform()
        self.data_loaded = True


class OpenSLR52SpeakerClustering(_LocalDatasetMixin, AbsTaskClustering):
    local_dir_name = "openslr52_speaker"

    metadata = mteb.TaskMetadata(
        name="OpenSLR52SpeakerClustering",
        description=(
            "Sinhala speaker clustering. Given read-speech utterances from the "
            "OpenSLR-52 Sinhala ASR corpus, cluster them by speaker without "
            "access to speaker labels."
        ),
        reference="https://openslr.org/52/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR52-Speaker",
            # Replace after running scripts/prepare/prepare_openslr_speaker.py.
            "revision": "main",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="v_measure",
        date=("2018-06-28", "2018-06-28"),
        domains=["Spoken"],
        # MTEB has no "Speaker Clustering" task_subtype literal (it offers
        # Gender/Accent/Emotion/Music/Vehicle clustering only), so the field is
        # left unset rather than mislabelled.
        task_subtypes=None,
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

    # the corpus is large; the prepared dataset is already subsampled
    max_document_to_embed = None
    max_fraction_of_documents_to_embed = None


class OpenSLR30SpeakerClustering(_LocalDatasetMixin, AbsTaskClustering):
    local_dir_name = "openslr30_speaker"

    metadata = mteb.TaskMetadata(
        name="OpenSLR30SpeakerClustering",
        description=(
            "Sinhala speaker clustering on the OpenSLR-30 crowd-sourced "
            "multi-speaker corpus. Twelve speakers recorded at 48 kHz, giving a "
            "cleaner and much smaller contrast to OpenSLR-52."
        ),
        reference="https://openslr.org/30/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR30-Speaker",
            # Replace after running scripts/prepare/prepare_openslr_speaker.py.
            "revision": "main",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="v_measure",
        date=("2017-10-03", "2017-10-03"),
        domains=["Spoken"],
        # MTEB has no "Speaker Clustering" task_subtype literal (it offers
        # Gender/Accent/Emotion/Music/Vehicle clustering only), so the field is
        # left unset rather than mislabelled.
        task_subtypes=None,
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

    max_document_to_embed = None
    max_fraction_of_documents_to_embed = None
