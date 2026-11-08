#!/bin/bash
# Re-evaluate ONLY the two mis-loaded checkpoints, with the validating wrapper,
# into NEW cache directories. The original caches are never written.
#
#   jobs/rerun_corrected.sh <gpu> <model>
#
# SIMTEB_CORRECTED_LOADERS=1 swaps in ValidatedWav2Vec2Wrapper, which resolves
# the architecture from the checkpoint config and raises rather than returning a
# partially loaded backbone.
set -uo pipefail

GPU="${1:?usage: rerun_corrected.sh <gpu> <model>}"
MODEL="${2:?usage: rerun_corrected.sh <gpu> <model>}"

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT/simteb_audio" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/simteb_audio/src"
export SIMTEB_AUDIO_LOCAL="$ROOT/simteb_audio/data_run"
export SIMTEB_CORRECTED_LOADERS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

SIN_CACHE="$ROOT/simteb_audio/mteb_cache_corrected"
MAEB_CACHE="$ROOT/simteb_audio/maeb_cache_corrected"
mkdir -p "$SIN_CACHE" "$MAEB_CACHE"

echo "=================================================================="
echo "corrected rerun | gpu $GPU | $MODEL"
echo "started $(date)"
echo "  SinAEB cache: $SIN_CACHE"
echo "  MAEB   cache: $MAEB_CACHE"
echo "=================================================================="

echo; echo ">>> SinAEB ($(date +%T))"
MTEB_CACHE="$SIN_CACHE" python evaluate.py --model "$MODEL" \
  --cache-folder "$SIN_CACHE" 2>&1 | tail -60

echo; echo ">>> MAEB ($(date +%T))"
MTEB_CACHE="$MAEB_CACHE" python maeb_evaluate.py --model "$MODEL" \
  --cache-folder "$MAEB_CACHE" 2>&1 | tail -60

safe="${MODEL//\//__}"
nsin=$(find "$SIN_CACHE/results/$safe" -name '*.json' ! -name 'model_meta.json' \
       2>/dev/null | wc -l)
nmaeb=$(find "$MAEB_CACHE/results/$safe" -name '*.json' ! -name 'model_meta.json' \
        2>/dev/null | wc -l)
echo; echo "cells: SinAEB $nsin/14  MAEB $nmaeb/19"
echo "gpu $GPU finished $MODEL $(date)"
