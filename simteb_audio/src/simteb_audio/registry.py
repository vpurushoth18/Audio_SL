from simteb_audio.tasks.classification import (
    OpenSLR30SpeakerClassification,
    OpenSLR52SpeakerClassification,
)
from simteb_audio.tasks.clustering import (
    OpenSLR30SpeakerClustering,
    OpenSLR52SpeakerClustering,
    WorldSpeechSinhalaQualityClassification,
    WorldSpeechSinhalaSessionClustering,
)
from simteb_audio.tasks.pair_classification import (
    OpenSLR30SpeakerPairClassification,
    OpenSLR52SpeakerPairClassification,
)
from simteb_audio.tasks.retrieval import (
    OpenSLR52A2TRetrieval,
    OpenSLR52AudioReranking,
    OpenSLR52AudioRetrieval,
    OpenSLR52T2ARetrieval,
)

TASK_REGISTRY = {
    "OpenSLR52SpeakerClustering": OpenSLR52SpeakerClustering,
    "OpenSLR30SpeakerClustering": OpenSLR30SpeakerClustering,
    "OpenSLR52SpeakerClassification": OpenSLR52SpeakerClassification,
    "OpenSLR30SpeakerClassification": OpenSLR30SpeakerClassification,
    "OpenSLR52SpeakerPairClassification": OpenSLR52SpeakerPairClassification,
    "OpenSLR30SpeakerPairClassification": OpenSLR30SpeakerPairClassification,
    "OpenSLR52AudioRetrieval": OpenSLR52AudioRetrieval,
    "OpenSLR52AudioReranking": OpenSLR52AudioReranking,
    "WorldSpeechSinhalaSessionClustering": WorldSpeechSinhalaSessionClustering,
    "WorldSpeechSinhalaQualityClassification": WorldSpeechSinhalaQualityClassification,
    "OpenSLR52A2TRetrieval": OpenSLR52A2TRetrieval,
    "OpenSLR52T2ARetrieval": OpenSLR52T2ARetrieval,
}

CATEGORY_REGISTRY = {
    "Clustering": [
        "OpenSLR52SpeakerClustering",
        "OpenSLR30SpeakerClustering",
        "WorldSpeechSinhalaSessionClustering",
    ],
    "Classification": [
        "OpenSLR52SpeakerClassification",
        "OpenSLR30SpeakerClassification",
        "WorldSpeechSinhalaQualityClassification",
    ],
    "PairClassification": [
        "OpenSLR52SpeakerPairClassification",
        "OpenSLR30SpeakerPairClassification",
    ],
    "Retrieval": ["OpenSLR52AudioRetrieval"],
    "Reranking": ["OpenSLR52AudioReranking"],
    # need a model that embeds audio AND text into one space
    "CrossModal": ["OpenSLR52A2TRetrieval", "OpenSLR52T2ARetrieval"],
}

# tasks runnable with an audio-only encoder
AUDIO_ONLY_TASKS = [
    n for n in TASK_REGISTRY
    if n not in CATEGORY_REGISTRY["CrossModal"]
]


def get_task(name):
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task: {name}. Available tasks: {list(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name]()


def get_tasks(names=None, categories=None):
    if names is not None:
        return [get_task(n) for n in names]
    if categories is not None:
        selected = []
        for c in categories:
            if c not in CATEGORY_REGISTRY:
                raise KeyError(f"Unknown category: {c}")
            selected.extend(CATEGORY_REGISTRY[c])
        return [get_task(n) for n in dict.fromkeys(selected)]
    return [cls() for cls in TASK_REGISTRY.values()]
