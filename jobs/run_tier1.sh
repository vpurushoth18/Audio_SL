#!/bin/bash
# Tier 1 models (permissive licences only) across every audio-only task.
#
#   setsid nohup jobs/run_tier1.sh > logs/run_tier1.log 2>&1 < /dev/null &
#
# Ordered cheapest first, so partial results are still a usable table if the
# run is interrupted. A model that fails is logged and skipped.

set -uo pipefail

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT/simteb_audio" || exit 1

export PYTHONPATH="$ROOT/simteb_audio/src"
export SIMTEB_AUDIO_LOCAL="$ROOT/simteb_audio/data_run"
export MTEB_CACHE="$ROOT/simteb_audio/mteb_cache"
export TOKENIZERS_PARALLELISM=false

# cheapest first; all MIT or Apache-2.0
MODELS=(
  "openai/whisper-tiny"                 # MIT      0.04B
  "openai/whisper-base"                 # MIT      0.07B
  "openai/whisper-small"                # MIT      0.24B
  "microsoft/wavlm-large"               # MIT      0.32B  English-only control
  "facebook/hubert-large-ls960-ft"      # MIT      0.32B  English-only control
  "openai/whisper-medium"               # MIT      0.77B
  "facebook/wav2vec2-xls-r-1b"          # Apache   1.00B
  "facebook/wav2vec2-xls-r-2b"          # Apache   2.00B
)

# already scored on clustering, still need the newer task types
BACKFILL=(
  "facebook/wav2vec2-xls-r-300m"
  "facebook/mms-300m"
  "facebook/w2v-bert-2.0"
  "openai/whisper-large-v3"
)

TASKS=(
  "OpenSLR52SpeakerClustering"
  "OpenSLR30SpeakerClustering"
  "OpenSLR52SpeakerClassification"
  "OpenSLR30SpeakerClassification"
  "OpenSLR52SpeakerPairClassification"
  "OpenSLR30SpeakerPairClassification"
  "OpenSLR52AudioRetrieval"
  "OpenSLR52AudioReranking"
)

echo "=================================================================="
echo "host $(hostname)  pid $$  started $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "models: ${#MODELS[@]} new + ${#BACKFILL[@]} backfill | tasks: ${#TASKS[@]}"
echo "=================================================================="

run_grid () {
  local label="$1"; shift
  for model in "$@"; do
    for task in "${TASKS[@]}"; do
      echo
      echo ">>> [$label] $model :: $task   ($(date +%T))"
      python evaluate.py --model "$model" --tasks "$task" --batch-size 8 \
          2>&1 | grep -avE "it/s\]|s/it\]|\[A"
      rc=${PIPESTATUS[0]}
      [ "$rc" -ne 0 ] && echo "!!! FAILED: $model :: $task (exit $rc)"
    done
  done
}

# backfill first: those models already work, so the new task types get
# validated before spending hours on the larger models
run_grid backfill "${BACKFILL[@]}"
run_grid new "${MODELS[@]}"

echo
echo "[collect]"
python "$ROOT/eval/collect_mteb_results.py" \
    --cache "$MTEB_CACHE" --out "$ROOT/eval/simteb_audio_results"

echo
echo "finished $(date)"
