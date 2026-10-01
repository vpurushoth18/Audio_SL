#!/bin/bash
# One GPU worker. Runs every task for the models named on the command line.
#
#   jobs/worker.sh <gpu_id> <model> [<model> ...]
#
# Launch two disjoint halves, one per GPU:
#   setsid nohup jobs/worker.sh 0 modelA modelB > logs/gpu0.log 2>&1 < /dev/null &
#   setsid nohup jobs/worker.sh 1 modelC modelD > logs/gpu1.log 2>&1 < /dev/null &
#
# Cells already in the MTEB cache are skipped by mteb.evaluate's default
# "only-missing" strategy, so re-running a worker is cheap and safe.

set -uo pipefail

GPU="${1:?usage: worker.sh <gpu_id> <model>...}"
shift
MODELS=("$@")

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT/simteb_audio" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/simteb_audio/src"
export SIMTEB_AUDIO_LOCAL="$ROOT/simteb_audio/data_run"
export MTEB_CACHE="$ROOT/simteb_audio/mteb_cache"
export TOKENIZERS_PARALLELISM=false
# sklearn probes are single-threaded per fit; keep BLAS from oversubscribing
# now that two workers share the box
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

TASKS=(
  "OpenSLR52SpeakerClustering"
  "OpenSLR30SpeakerClustering"
  "WorldSpeechSinhalaSessionClustering"
  "OpenSLR52SpeakerClassification"
  "OpenSLR30SpeakerClassification"
  "WorldSpeechSinhalaQualityClassification"
  "OpenSLR52SpeakerPairClassification"
  "OpenSLR30SpeakerPairClassification"
  "OpenSLR52AudioRetrieval"
  "OpenSLR52AudioReranking"
)

echo "=================================================================="
echo "GPU $GPU | pid $$ | $(hostname) | started $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    -i "$GPU" 2>/dev/null
echo "models (${#MODELS[@]}): ${MODELS[*]}"
echo "tasks: ${#TASKS[@]}"
echo "=================================================================="

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo
    echo ">>> [gpu$GPU] $model :: $task   ($(date +%T))"
    python evaluate.py --model "$model" --tasks "$task" --batch-size 8 \
        2>&1 | grep -avE "it/s\]|s/it\]|\[A"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && echo "!!! FAILED: $model :: $task (exit $rc)"
  done
done

echo
echo "GPU $GPU finished $(date)"
