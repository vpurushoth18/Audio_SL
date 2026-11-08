#!/bin/bash
# Full SiMTEB-Audio evaluation: build the runnable datasets, evaluate every
# model on every task, collect the results table.
#
# Launched detached so it survives SSH disconnect:
#   setsid nohup jobs/run_all.sh > logs/run_all.log 2>&1 < /dev/null &
#
# Watch:  tail -f logs/run_all.log
# Stop:   pkill -f run_all.sh

set -uo pipefail

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT/simteb_audio" || exit 1

export PYTHONPATH="$ROOT/simteb_audio/src"
export SIMTEB_AUDIO_LOCAL="$ROOT/simteb_audio/data_run"
export MTEB_CACHE="$ROOT/simteb_audio/mteb_cache"
export TOKENIZERS_PARALLELISM=false

MODELS=(
  "facebook/wav2vec2-xls-r-300m"
  "facebook/mms-300m"
  "facebook/w2v-bert-2.0"
  "openai/whisper-large-v3"
)
TASKS=(
  "OpenSLR52SpeakerClustering"
  "OpenSLR30SpeakerClustering"
)

echo "=================================================================="
echo "host $(hostname)   pid $$   started $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=================================================================="

# ---- 1. datasets the evaluation can decode ----------------------------
# The copies in simteb_audio/data hold encoded bytes for the Hub and need
# torchcodec + FFmpeg to decode, which this machine lacks. The array encoding
# stores decoded audio, so it runs here.
if [ -d "$SIMTEB_AUDIO_LOCAL/openslr52_speaker" ] && \
   [ -d "$SIMTEB_AUDIO_LOCAL/openslr30_speaker" ]; then
  echo "[1/3] runnable datasets present, skipping build"
else
  echo "[1/3] building runnable datasets (array encoding)"
  python scripts/prepare/prepare_openslr_speaker.py \
      --datasets openslr52 openslr30 \
      --encoding array \
      --out "$SIMTEB_AUDIO_LOCAL" 2>&1 | grep -vE "examples/s|Saving the dataset"
  if [ ! -d "$SIMTEB_AUDIO_LOCAL/openslr52_speaker" ]; then
    echo "FATAL: dataset build failed"; exit 1
  fi
fi

# ---- 2. the model x task grid -----------------------------------------
echo
echo "[2/3] evaluating ${#MODELS[@]} models x ${#TASKS[@]} tasks"
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo
    echo "------------------------------------------------------------------"
    echo ">>> $model :: $task   ($(date +%T))"
    echo "------------------------------------------------------------------"
    python evaluate.py --model "$model" --tasks "$task" --batch-size 8 \
        2>&1 | grep -vE "it/s\]|s/it\]|it/s\]\[A"
    rc=${PIPESTATUS[0]}
    # one model failing must not abandon the rest of the grid
    [ "$rc" -ne 0 ] && echo "!!! FAILED: $model on $task (exit $rc) - continuing"
  done
done

# ---- 3. collect ---------------------------------------------------------
echo
echo "[3/3] collecting results"
python "$ROOT/eval/collect_mteb_results.py" \
    --cache "$MTEB_CACHE" \
    --out "$ROOT/eval/simteb_audio_results"

echo
echo "=================================================================="
echo "finished $(date)"
echo "=================================================================="
