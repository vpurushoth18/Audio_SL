"""Sinhala video clustering on VoxLingua107.

VoxLingua107 is automatically harvested YouTube audio, so the Sinhala subset is
the most domain-diverse Sinhala speech available here - unlike OpenSLR's read
prompts or WorldSpeech's single parliament.

The only label the corpus carries is `lang: si`, which is constant across the
subset and so cannot define clusters. What is usable is the source video: each
clip id is `<youtubeID>__U__S<start>-<end>`, so clips can be grouped by the video
they were cut from. Clips of one video share a speaker, a channel and an
acoustic setting, which makes this a session-structure task over genuinely
varied material.

Labels are automatic, not human-verified: VoxLingua107's language tags come from
a pipeline, and a video may contain more than one speaker. Treat the grouping as
"same recording", not "same speaker".
"""

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClustering

from .._local import LocalDatasetMixin

CITATION = r"""
@inproceedings{valk2021voxlingua107,
  author = {Valk, Jörgen and Alumäe, Tanel},
  booktitle = {2021 IEEE Spoken Language Technology Workshop (SLT)},
  organization = {IEEE},
  pages = {652--658},
  title = {{VoxLingua107: a Dataset for Spoken Language Recognition}},
  year = {2021},
}
"""


class VoxLingua107SinhalaVideoClustering(LocalDatasetMixin, AbsTaskClustering):
    local_dir_name = "voxlingua_sin_video"

    metadata = TaskMetadata(
        name="VoxLingua107SinhalaVideoClustering",
        description=(
            "Cluster Sinhala speech clips from VoxLingua107 by the YouTube video "
            "they were extracted from. Automatically harvested in-the-wild audio, "
            "so this measures recording and channel structure over far more "
            "varied material than read-speech corpora."
        ),
        reference="https://bark.phon.ioc.ee/voxlingua107/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-VoxLingua107-Sinhala",
            "revision": "main",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score="v_measure",
        date=("2020-01-01", "2021-12-31"),
        domains=["Spoken", "Web"],
        license="cc-by-4.0",
        annotations_creators="automatic",
        dialect=[],
        sample_creation="found",
        bibtex_citation=CITATION,
    )

    input_column_name = "audio"
    label_column_name = "video_id"
