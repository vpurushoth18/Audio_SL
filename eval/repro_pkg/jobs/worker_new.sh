#!/bin/bash
# One GPU worker for the four NEW tasks (Omnilingual + VoxLingua107).
#
#   jobs/worker_new.sh <gpu_id> <model> [<model> ...]
#
# Separate from worker.sh so the 120 existing cells are not re-walked. Cells
# already in the MTEB cache are skipped regardless.

set -uo pipefail

GPU="${1:?usage: worker_new.sh <gpu_id> <model>...}"
shift
MODELS=("$@")

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT/simteb_audio" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/simteb_audio/src"
export SIMTEB_AUDIO_LOCAL="$ROOT/simteb_audio/data_run"
export MTEB_CACHE="$ROOT/simteb_audio/mteb_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

TASKS=(
  "OmnilingualSinhalaSpeakerClustering"
  "OmnilingualSinhalaSpeakerClassification"
  "OmnilingualSinhalaContentRetrieval"
  "VoxLingua107SinhalaVideoClustering"
)

echo "=================================================================="
echo "GPU $GPU | pid $$ | $(hostname) | started $(date)"
echo "models (${#MODELS[@]}): ${MODELS[*]}"
echo "tasks (${#TASKS[@]}): ${TASKS[*]}"
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
