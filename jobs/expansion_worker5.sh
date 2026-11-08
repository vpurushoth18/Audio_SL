#!/bin/bash
# One GPU's share of the expansion. Model-complete, resumable, stoppable.
#
#   jobs/expansion_worker.sh <gpu_id> <stride_offset> <stride> [max_wave]
#
# Two workers with offsets 0 and 1 and stride 2 split the list between the GPUs
# while keeping the wave order, so stopping early still leaves a sample spread
# across families rather than six sizes of one model.
#
# A checkpoint is complete only when BOTH benchmarks have produced results for
# it. A model present on one side cannot enter a cross-benchmark correlation,
# so finishing tasks in model order is what makes an early stop usable.
#
# Stop cleanly:  touch logs/STOP_EXPANSION
#                the checkpoint in flight finishes, then the worker exits.

set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1
export PYTHONPATH="$R/simteb_audio/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

GPU="${1:?usage: expansion_worker.sh <gpu> <offset> <stride> [max_wave]}"
OFFSET="${2:-0}"; STRIDE="${3:-1}"; MAXWAVE="${4:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"

STOP="logs/STOP_EXPANSION"
DONE="logs/expansion_done_wave5.txt"
touch "$DONE"

mapfile -t PLAN < <(python - "$MAXWAVE" "$OFFSET" "$STRIDE" <<'PY'
import sys, pandas as pd
mx, off, stride = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
d = pd.read_csv("eval/analysis/expansion_inventory.tsv", sep="\t")
d = d[d.wave == mx].sort_values(["wave", "family", "model"])   # wave 5 ONLY
for i, (_, r) in enumerate(d.iterrows()):
    if i % stride == off:
        print(r.model)
PY
)

echo "=================================================================="
echo "expansion worker | gpu $GPU | ${#PLAN[@]} checkpoints | waves 1-$MAXWAVE"
echo "started $(date) on $(hostname)"
echo "=================================================================="

n=0
for model in "${PLAN[@]}"; do
  [ -f "$STOP" ] && { echo; echo ">>> stop requested after $n checkpoints"; break; }
  grep -qxF "$model" "$DONE" 2>/dev/null && { echo ">>> $model done, skip"; continue; }

  safe="${model//\//__}"
  echo; echo "------------------------------------------------------------------"
  echo ">>> [gpu$GPU $((n+1))/${#PLAN[@]}] $model   ($(date +%T))"
  echo "------------------------------------------------------------------"

  ./jobs/worker.sh      "$GPU" "$model" 2>&1 | tail -40
  ./jobs/worker_new.sh  "$GPU" "$model" 2>&1 | tail -20
  ./jobs/worker_maeb.sh "$GPU" "$model" 2>&1 | tail -40

  nsin=$(find "simteb_audio/mteb_cache/results/$safe" -name '*.json' \
         ! -name 'model_meta.json' 2>/dev/null | wc -l)
  nmaeb=$(find "simteb_audio/maeb_cache/results/$safe" -name '*.json' \
          ! -name 'model_meta.json' 2>/dev/null | wc -l)
  echo "    cells: SinAEB $nsin/14  MAEB $nmaeb/19"

  if [ "$nsin" -ge 12 ] && [ "$nmaeb" -ge 16 ]; then
    echo "$model" >> "$DONE"; n=$((n+1))
    echo "    COMPLETE  ($n this worker)"
    # weights are needed once; drop them so peak disk stays near one model
    hub="$HOME/.cache/huggingface/hub/models--${model//\//--}"
    if [ -d "$hub" ]; then
      sz=$(du -sh "$hub" 2>/dev/null | cut -f1)
      rm -rf "$hub" && echo "    freed $sz of weights"
    fi
    flock -w 300 logs/.analysis.lock \
      python -u eval/expansion_analysis.py > logs/expansion_analysis.log 2>&1 \
      && echo "    analysis refreshed" || echo "    analysis skipped"
  else
    echo "    INCOMPLETE - not recorded, will retry on resume"
  fi
done

echo; echo "gpu $GPU finished $(date), $n checkpoints this run"
