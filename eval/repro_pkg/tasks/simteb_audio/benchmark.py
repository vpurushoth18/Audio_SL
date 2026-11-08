from simteb_audio.registry import AUDIO_ONLY_TASKS, get_tasks

# the default benchmark: everything an audio-only encoder can run
SIMTEB_AUDIO_TASKS = AUDIO_ONLY_TASKS


def get_benchmark_tasks():
    return get_tasks(names=SIMTEB_AUDIO_TASKS)
