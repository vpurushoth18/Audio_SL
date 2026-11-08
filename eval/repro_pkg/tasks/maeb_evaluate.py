#!/usr/bin/env python3
"""Evaluate a model on MAEB(beta, audio-only) - MTEB's own audio benchmark.

Used for the external-validity comparison: does a model's Sinhala ranking match
its ranking on a general multilingual audio benchmark?

Scoring, tasks and metrics are MTEB's. The only local change is
`sf_audio_patch`, which decodes HF Audio columns with soundfile because
torchcodec cannot be installed on this cluster.

    python maeb_evaluate.py --model openai/whisper-tiny
    python maeb_evaluate.py --model openai/whisper-tiny --tasks BeijingOpera
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import simteb_audio.sf_audio_patch as sf_patch  # noqa: E402  must precede data loads

import mteb  # noqa: E402

import simteb_audio.models as extra_models  # noqa: E402

BENCHMARK = "MAEB(beta, audio-only)"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a model on MAEB audio-only.")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", nargs="*", default=None,
                   help="subset of MAEB task names; omit for all of them")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-folder", default=None)
    p.add_argument("--list-tasks", action="store_true")
    return p.parse_args()


def maeb_tasks():
    bench = next(b for b in mteb.get_benchmarks() if b.name == BENCHMARK)
    return list(bench.tasks)


def main():
    a = parse_args()

    if a.list_tasks:
        for t in sorted(maeb_tasks(), key=lambda x: x.metadata.name):
            print(f"{t.metadata.name:36s} {t.metadata.type:26s} "
                  f"{t.metadata.main_score}")
        return 0

    cache = Path(a.cache_folder or os.environ.get("MTEB_CACHE")
                 or (Path.home() / ".cache" / "mteb")).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    tasks = maeb_tasks()
    if a.tasks:
        want = set(a.tasks)
        tasks = [t for t in tasks if t.metadata.name in want]
        missing = want - {t.metadata.name for t in tasks}
        if missing:
            print(f"not in {BENCHMARK}: {sorted(missing)}", file=sys.stderr)
            return 1

    print("=" * 70)
    print(f"MAEB: {BENCHMARK}")
    print("=" * 70)
    print(f"Model:        {a.model}")
    print(f"Batch size:   {a.batch_size}")
    print(f"MTEB cache:   {cache}")
    print(f"Audio patch:  {'soundfile (torchcodec absent)' if sf_patch._APPLIED else 'torchcodec'}")
    print(f"Registered:   {', '.join(extra_models.register()) or 'already registered'}")
    print(f"Tasks:        {len(tasks)}")
    print("=" * 70)

    model = mteb.get_model(a.model)
    results = mteb.evaluate(
        model, tasks=tasks, cache=mteb.ResultCache(cache_path=cache),
        encode_kwargs={"batch_size": a.batch_size},
        raise_error=False,          # one bad task must not abandon the rest
    )

    print("\n" + "=" * 70)
    print("MAEB results")
    print("=" * 70)
    for r in results:
        try:
            print(f"{r.task_name}: {r.get_score():.4f}")
        except Exception:
            print(r)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
