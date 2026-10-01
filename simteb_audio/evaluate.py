#!/usr/bin/env python3
"""Evaluate an audio embedding model on the Sinhala audio tasks."""

import argparse
import os
import sys
from pathlib import Path

import mteb

sys.path.insert(0, str(Path(__file__).parent / "src"))

from simteb_audio.benchmark import get_benchmark_tasks  # noqa: E402
from simteb_audio.models import register as register_models  # noqa: E402
from simteb_audio.registry import get_tasks  # noqa: E402

# adds mms-300m and w2v-bert-2.0, which MTEB does not ship
_ADDED = register_models()


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate an audio embedding model on SiMTEB-Audio."
    )
    p.add_argument("--model", required=True,
                   help="Hugging Face or MTEB model name, or a local path")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="individual task names; omit to run the full benchmark")
    p.add_argument("--categories", nargs="*", default=None,
                   help="task categories, e.g. Clustering")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cache-folder", default=None,
                   help="MTEB result cache; falls back to MTEB_CACHE, "
                        "then ~/.cache/mteb")
    return p.parse_args()


def get_cache_folder(args):
    if args.cache_folder:
        folder = Path(args.cache_folder)
    elif os.environ.get("MTEB_CACHE"):
        folder = Path(os.environ["MTEB_CACHE"])
    else:
        folder = Path.home() / ".cache" / "mteb"
    folder = folder.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def select_tasks(args):
    if args.tasks and args.categories:
        raise ValueError("Specify either --tasks or --categories, not both.")
    if args.tasks:
        return get_tasks(names=args.tasks)
    if args.categories:
        return get_tasks(categories=args.categories)
    return get_benchmark_tasks()


def main():
    args = parse_args()
    cache_folder = get_cache_folder(args)
    local = os.environ.get("SIMTEB_AUDIO_LOCAL")

    print("=" * 70)
    print("SiMTEB-Audio")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Batch size:   {args.batch_size}")
    print(f"MTEB cache:   {cache_folder}")
    print(f"Local data:   {local or 'not set (loading from the Hub)'}")
    if _ADDED:
        print(f"Registered:   {', '.join(_ADDED)}")
    print("=" * 70)

    print(f"\nLoading model: {args.model}")
    model = mteb.get_model(args.model)

    tasks = select_tasks(args)
    print("\nTasks:")
    for task in tasks:
        print(f"  - {task.metadata.name}")

    cache = mteb.ResultCache(cache_path=cache_folder)

    print("\nStarting evaluation...\n")
    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        encode_kwargs={"batch_size": args.batch_size},
    )

    print("\n" + "=" * 70)
    print("SiMTEB-Audio Results")
    print("=" * 70)
    for result in results:
        try:
            print(f"{result.task_name}: {result.get_score():.4f}")
        except Exception:
            print(result)
    print("=" * 70)
    print(f"Full MTEB result files are stored in:\n{cache_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
