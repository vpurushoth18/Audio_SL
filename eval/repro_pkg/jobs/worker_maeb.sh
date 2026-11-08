#!/bin/bash
# One GPU worker for MAEB(beta, audio-only) - 19 tasks.
#
#   jobs/worker_maeb.sh <gpu_id> <model> [<model> ...]
#
# Runs one task at a time so a failure costs one cell, not the whole model.
# Cells already in the MTEB cache are skipped.

set -uo pipefail

GPU="${1:?usage: worker_maeb.sh <gpu_id> <model>...}"
shift
MODELS=("$@")

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT/simteb_audio" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/simteb_audio/src"
export MTEB_CACHE="$ROOT/simteb_audio/maeb_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# MAEB(beta, audio-only), smallest datasets first so partial results are usable
TASKS=(
  "BeijingOpera"
  "MridinghamTonic"
  "VoxPopuliGenderClustering"
  "VoxPopuliLanguageID"
  "BirdCLEF"
  "VoxCelebSA"
  "CREMA_D"
  "CREMA_DClustering"
  "CommonLanguageAgeDetection"
  "VehicleSoundClustering"
  "MInDS14"
  "NMSQAPairClassification"
  "CREMADPairClassification"
  "GTZANAudioReranking"
  "GTZANGenre"
  "IEMOCAPGender"
  "JamAltArtistA2ARetrieval"
  "SIBFLEURS"
  "VoxPopuliAccentPairClassification"
)

echo "=================================================================="
echo "MAEB | GPU $GPU | pid $$ | $(hostname) | started $(date)"
echo "models (${#MODELS[@]}): ${MODELS[*]}"
echo "tasks: ${#TASKS[@]}"
echo "=================================================================="

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo
    echo ">>> [maeb gpu$GPU] $model :: $task   ($(date +%T))"
    python maeb_evaluate.py --model "$model" --tasks "$task" --batch-size 8 \
        2>&1 | grep -avE "it/s\]|s/it\]|\[A|is currently in beta|UserWarning|super\(\)|task = _TASKS|ConvergenceWarning|n_iter_i|scikit-learn.org|Increase the number|You might also|Please also refer"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && echo "!!! FAILED: $model :: $task (exit $rc)"
  done
done

echo
echo "MAEB GPU $GPU finished $(date)"
