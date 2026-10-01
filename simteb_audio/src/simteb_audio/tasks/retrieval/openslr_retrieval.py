"""Retrieval and reranking tasks on OpenSLR-52.

Four tasks, all from the same prepared dataset, which pairs each utterance with
its transcript and its content group (utterances sharing a transcript, read by
different speakers):

  A2T   audio query  -> transcript document      speech/text alignment
  T2A   text query   -> audio document           cross-modal retrieval
  A2A   audio query  -> audio of the same sentence, by a DIFFERENT speaker
  Rerank  audio query -> a small candidate pool of positives and hard negatives

MTEB implements audio reranking as a retrieval task over a per-query candidate
pool (see FSDnoisy18kAudioReranking), so all four subclass AbsTaskRetrieval and
differ only in how corpus, queries and qrels are built.

A2T and T2A need a model that embeds BOTH audio and text into one space. The
audio-only encoders evaluated so far cannot run them; they are for CLAP-style
joint models. A2A and reranking are audio-only and run with any audio encoder.
"""

import os
from collections import defaultdict
from typing import Any

import mteb
from datasets import Dataset, DatasetDict, load_from_disk
from mteb.abstasks import AbsTaskRetrieval

_CITATION = r"""
@inproceedings{kjartansson-etal-sltu2018,
  address = {Gurugram, India},
  author = {Kjartansson, Oddur and Sarin, Supheakmungkol and Pipatsrisawat, Knot and Jansche, Martin and Ha, Linne},
  booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  month = {August},
  pages = {52--55},
  title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali}},
  year = {2018},
}
"""


def _meta(name, description, category, modalities, main_score, subtype):
    return mteb.TaskMetadata(
        name=name,
        description=description,
        reference="https://openslr.org/52/",
        dataset={
            "path": "Sinhala-NLP/SiMTEB-Audio-OpenSLR52-Retrieval",
            "revision": "main",
        },
        type="Any2AnyRetrieval",
        category=category,
        modalities=modalities,
        eval_splits=["test"],
        eval_langs=["sin-Sinh"],
        main_score=main_score,
        date=("2018-06-28", "2018-06-28"),
        domains=["Spoken"],
        task_subtypes=[subtype] if subtype else None,
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_CITATION,
    )


class _OpenSLR52RetrievalBase(AbsTaskRetrieval):
    """Shared loading. Subclasses only implement _build."""

    local_dir_name = "openslr52_retrieval"

    def _source(self) -> Dataset:
        root = os.environ.get("SIMTEB_AUDIO_LOCAL")
        if root:
            return load_from_disk(os.path.join(root, self.local_dir_name))["test"]
        import datasets as ds

        return ds.load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split="test",
        )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        # These tasks are monolingual (eval_langs is a list, so is_multilingual
        # is False), and MTEB then expects queries/corpus keyed by SPLIT only.
        # The extra subset level used by multilingual tasks like FLEURS makes
        # convert_v1_dataset_format_to_v2 hand a Dataset to pyarrow.
        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        src = self._source()
        split = self.metadata.eval_splits[0]
        queries, corpus, qrels = self._build(src)
        self.queries[split] = queries
        self.corpus[split] = corpus
        self.relevant_docs[split] = qrels

    def _build(self, src):
        raise NotImplementedError


class OpenSLR52A2TRetrieval(_OpenSLR52RetrievalBase):
    metadata = _meta(
        "OpenSLR52A2TRetrieval",
        "Sinhala speech-to-text retrieval. Given a spoken utterance from the "
        "OpenSLR-52 corpus, retrieve its transcript from a corpus of "
        "transcripts. Requires a model that embeds audio and text jointly.",
        category="a2t",
        modalities=["audio", "text"],
        main_score="hit_rate_at_5",
        subtype="Speech Transcription Retrieval",
    )

    def _build(self, src):
        # one document per unique transcript, first-seen order
        text_of: dict[str, str] = {}
        for sid, text in zip(src["sentence_id"], src["transcript"]):
            text_of.setdefault(sid, text)

        queries = src.select_columns(["audio"]).add_column("id", src["utt_id"])
        corpus = Dataset.from_dict(
            {"id": list(text_of), "text": list(text_of.values())}
        )
        # identical text under different ids is indistinguishable, so both count
        ids_of_text: dict[str, list[str]] = defaultdict(list)
        for sid, text in text_of.items():
            ids_of_text[text].append(sid)
        qrels = {
            utt: dict.fromkeys(ids_of_text[text], 1)
            for utt, text in zip(src["utt_id"], src["transcript"])
        }
        return queries, corpus, qrels


class OpenSLR52T2ARetrieval(_OpenSLR52RetrievalBase):
    metadata = _meta(
        "OpenSLR52T2ARetrieval",
        "Sinhala text-to-speech retrieval. Given a transcript, retrieve the "
        "recordings of it. Every recording of the sentence counts as correct. "
        "Requires a model that embeds audio and text jointly.",
        category="t2a",
        modalities=["text", "audio"],
        main_score="hit_rate_at_5",
        subtype="Speech Retrieval",
    )

    def _build(self, src):
        text_of: dict[str, str] = {}
        for sid, text in zip(src["sentence_id"], src["transcript"]):
            text_of.setdefault(sid, text)

        queries = Dataset.from_dict(
            {"id": list(text_of), "text": list(text_of.values())}
        )
        corpus = src.select_columns(["audio"]).add_column("id", src["utt_id"])
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for sid, utt in zip(src["sentence_id"], src["utt_id"]):
            qrels[sid][utt] = 1
        return queries, corpus, dict(qrels)


class OpenSLR52AudioRetrieval(_OpenSLR52RetrievalBase):
    metadata = _meta(
        "OpenSLR52AudioRetrieval",
        "Sinhala audio-to-audio retrieval. Given a spoken utterance, retrieve "
        "other recordings of the SAME sentence spoken by DIFFERENT speakers. "
        "Recordings by the query's own speaker are not counted as correct, so "
        "the task cannot be solved by matching the voice.",
        category="a2a",
        modalities=["audio"],
        main_score="hit_rate_at_5",
        subtype="Speech Retrieval",
    )

    def _build(self, src):
        utt_ids = src["utt_id"]
        sids = src["sentence_id"]
        spks = src["speaker_id"]

        by_sentence: dict[str, list[int]] = defaultdict(list)
        for i, sid in enumerate(sids):
            by_sentence[sid].append(i)

        qrels: dict[str, dict[str, int]] = {}
        keep_q = []
        for i, (utt, sid, spk) in enumerate(zip(utt_ids, sids, spks)):
            # same sentence, different speaker
            rel = {
                utt_ids[j]: 1
                for j in by_sentence[sid]
                if j != i and spks[j] != spk
            }
            if rel:
                qrels[utt] = rel
                keep_q.append(i)

        corpus = src.select_columns(["audio"]).add_column("id", utt_ids)
        queries = (src.select(keep_q).select_columns(["audio"])
                      .add_column("id", [utt_ids[i] for i in keep_q]))
        return queries, corpus, qrels


class OpenSLR52AudioReranking(_OpenSLR52RetrievalBase):
    """Reranking as MTEB implements it: retrieval over a small candidate pool.

    Each query gets its cross-speaker positives plus hard negatives - other
    utterances by the SAME speaker, which a voice-matching model will rank
    highly and a content-aware model will not.
    """

    local_dir_name = "openslr52_rerank"

    metadata = _meta(
        "OpenSLR52AudioReranking",
        "Sinhala audio reranking. Given a spoken utterance, rank recordings of "
        "the same sentence by other speakers above hard negatives, which are "
        "other utterances by the query's own speaker. Rewards encoding what was "
        "said over who said it.",
        category="a2a",
        modalities=["audio"],
        main_score="map_at_5",
        subtype="Speech Retrieval",
    )

    def _build(self, src):
        utt_ids = src["utt_id"]
        corpus = src.select_columns(["audio"]).add_column("id", utt_ids)
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for q, pos in zip(src["query_id"], src["positive_ids"]):
            for p in pos:
                qrels[q][p] = 1
        q_idx = [i for i, u in enumerate(utt_ids) if u in qrels]
        queries = (src.select(q_idx).select_columns(["audio"])
                      .add_column("id", [utt_ids[i] for i in q_idx]))
        return queries, corpus, dict(qrels)
