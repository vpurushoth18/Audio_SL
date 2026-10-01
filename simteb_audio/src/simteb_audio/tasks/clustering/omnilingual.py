"""Sinhala tasks on Meta's Omnilingual ASR corpus.

A second read-speech corpus independent of OpenSLR, and the only Sinhala source
found so far that carries BOTH speaker ids and prompt ids - so the same
utterance-level structure the OpenSLR-52 tasks use is available here for
cross-corpus comparison.

Scale caveat, stated up front: the Sinhala subset has only **7 speakers** over
627 utterances. Speaker tasks with 7 classes are much easier than the 478-speaker
OpenSLR-52 versions, so scores are not comparable across the two corpora - read
them as a separate, easier column. The utterances are long (~69 s mean), and are
truncated to 20 s like everywhere else in this benchmark.

The 411 prompts include 158 read by more than one speaker, which is what makes
cross-speaker content retrieval possible.
"""

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification, AbsTaskClustering

from .._local import LocalDatasetMixin

CITATION = r"""
@misc{omnilingualasr2025,
  author = {{Meta AI}},
  title = {{Omnilingual ASR: a massively multilingual speech recognition corpus}},
  howpublished = {\url{https://huggingface.co/datasets/facebook/omnilingual-asr-corpus}},
  year = {2025},
}
"""


def _meta(name, description, task_type, main_score):
    return TaskMetadata(
        name=name,
        description=description,
        reference="https://huggingface.co/datasets/facebook/omnilingual-asr-corpus",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-Omnilingual-Sinhala",
            "revision": "main",
        },
        type=task_type,
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score=main_score,
        date=("2025-01-01", "2025-12-31"),
        domains=["Spoken"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=CITATION,
    )


class OmnilingualSinhalaSpeakerClustering(LocalDatasetMixin, AbsTaskClustering):
    local_dir_name = "omnilingual_sin_speaker"

    metadata = _meta(
        "OmnilingualSinhalaSpeakerClustering",
        "Cluster Sinhala read-speech utterances from Meta's Omnilingual ASR "
        "corpus by speaker. Only 7 speakers, so this is a much easier setting "
        "than the 478-speaker OpenSLR-52 task.",
        "AudioClustering",
        "v_measure",
    )

    input_column_name = "audio"
    label_column_name = "speaker_id"


class OmnilingualSinhalaSpeakerClassification(
    LocalDatasetMixin, AbsTaskClassification
):
    local_dir_name = "omnilingual_sin_speaker_clf"

    metadata = _meta(
        "OmnilingualSinhalaSpeakerClassification",
        "Identify which of 7 speakers produced a Sinhala utterance from Meta's "
        "Omnilingual ASR corpus.",
        "AudioClassification",
        "accuracy",
    )

    input_column_name = "audio"
    label_column_name = "speaker_id"
