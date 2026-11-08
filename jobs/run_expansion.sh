#!/bin/bash
# Expansion runner: model-complete, resumable, stoppable at any point.
#
#   setsid nohup jobs/run_expansion.sh > logs/expansion.log 2>&1 < /dev/null &
#
# The design question is "what do I have if I stop now?", so the unit of work
# is a CHECKPOINT, not a task. For each checkpoint it runs every SinAEB task
# and every MAEB task, then marks the checkpoint complete. Stopping after N
# checkpoints leaves N rows that can enter a cross-benchmark correlation,
# rather than 45 rows with holes in them.
#
# To stop cleanly:   touch logs/STOP_EXPANSION
# It finishes the checkpoint in flight, then exits. Nothing is left half-done.
#
# To resume:         rm logs/STOP_EXPANSION, launch again.
# Cells already in the MTEB cache are skipped, so resuming costs nothing.

set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1
export PYTHONPATH="$R/simteb_audio/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

GPU="${1:-0}"
MAX_WAVE="${2:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"

STOP="logs/STOP_EXPANSION"
DONE="logs/expansion_done.txt"
touch "$DONE"
rm -f "$STOP"

mapfile -t PLAN < <(python - "$MAX_WAVE" <<'PY'
import sys, pandas as pd
mx = int(sys.argv[1])
d = pd.read_csv("eval/analysis/expansion_inventory.tsv", sep="\t")
d = d[(d.wave > 0) & (d.wave <= mx)].sort_values(["wave", "family", "model"])
for _, r in d.iterrows():
    print(r.model)
PY
)

echo "=================================================================="
echo "expansion | gpu $GPU | waves 1-$MAX_WAVE | ${#PLAN[@]} new checkpoints"
echo "started $(date) on $(hostname)"
echo "stop cleanly with: touch $STOP"
echo "=================================================================="

n=0
for model in "${PLAN[@]}"; do
  if [ -f "$STOP" ]; then
    echo; echo ">>> stop requested, exiting after $n completed checkpoints"
    break
  fi
  if grep -qxF "$model" "$DONE" 2>/dev/null; then
    echo ">>> $model already complete, skipping"
    continue
  fi

  echo; echo "=================================================================="
  echo ">>> [$((n+1))/${#PLAN[@]}] $model   ($(date +%T))"
  echo "=================================================================="

  ok=1
  echo "--- SinAEB ---"
  if ! ./jobs/worker.sh "$GPU" "$model"; then ok=0; fi
  if ! ./jobs/worker_new.sh "$GPU" "$model"; then ok=0; fi
  echo "--- MAEB ---"
  if ! ./jobs/worker_maeb.sh "$GPU" "$model"; then ok=0; fi

  # a checkpoint counts as complete only if BOTH benchmarks produced results
  nsin=$(find simteb_audio/mteb_cache/results/"${model//\//__}" -name '*.json' \
         ! -name 'model_meta.json' 2>/dev/null | wc -l)
  nmaeb=$(find simteb_audio/maeb_cache/results/"${model//\//__}" -name '*.json' \
          ! -name 'model_meta.json' 2>/dev/null | wc -l)
  echo "    cells: SinAEB $nsin/14, MAEB $nmaeb/19"

  if [ "$nsin" -ge 12 ] && [ "$nmaeb" -ge 16 ]; then
    echo "$model" >> "$DONE"
    n=$((n+1))
    echo "    COMPLETE ($n checkpoints done this run)"
  else
    echo "    INCOMPLETE - not recorded; rerun will retry it"
  fi

  # refresh the analysis after every checkpoint, so results are always current
  python -u eval/expansion_analysis.py >> logs/expansion_analysis.log 2>&1 \
    && echo "    analysis refreshed" || echo "    analysis failed (continuing)"
done

echo
echo "finished $(date)"
echo "completed checkpoints this run: $n"
echo "total recorded complete: $(wc -l < "$DONE")"
date > logs/EXPANSION_STOPPED
