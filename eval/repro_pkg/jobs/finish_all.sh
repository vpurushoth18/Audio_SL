#!/bin/bash
# Wait for MAEB to finish, retry the cells that failed, then regenerate every
# downstream artefact. Detached and unattended:
#
#   setsid nohup jobs/finish_all.sh > logs/finish_all.log 2>&1 < /dev/null &
#
# Writes logs/FINISHED when the whole chain is done, so progress can be checked
# with a single test for that file.

set -uo pipefail
ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/simteb_audio/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

rm -f logs/FINISHED
echo "finisher started $(date), pid $$"

# 1. wait for both MAEB workers
while pgrep -f "worker_maeb.sh" >/dev/null; do
  n=$(( $(grep -ac '>>>' logs/maeb_gpu0.log 2>/dev/null) \
      + $(grep -ac '>>>' logs/maeb_gpu1.log 2>/dev/null) ))
  echo "  [$(date +%T)] MAEB $n/228"
  sleep 300
done
echo "MAEB workers exited $(date)"

# 2. retry the cells that failed, one at a time, on whichever GPU is free
echo
echo ">>> retrying failed MAEB cells"
grep -ah "!!! FAILED" logs/maeb_gpu*.log \
  | sed -E 's/.*FAILED: ([^ ]+) :: ([^ ]+).*/\1 \2/' | sort -u \
  | while read -r model task; do
      [ -z "${model:-}" ] && continue
      echo "  retry $model :: $task"
      ( cd simteb_audio && CUDA_VISIBLE_DEVICES=0 MTEB_CACHE="$ROOT/simteb_audio/maeb_cache" \
        python maeb_evaluate.py --model "$model" --tasks "$task" --batch-size 4 \
        2>&1 | tail -3 )
    done

# 3. the MAEB comparison, now over every model that completed
echo
echo ">>> MAEB vs Sinhala comparison"
python -u eval/maeb_compare.py

# 4. refresh the Sinhala tables and analyses
echo
echo ">>> refresh Sinhala results and analyses"
python -u eval/collect_mteb_results.py --cache "$ROOT/simteb_audio/mteb_cache" \
    --out "$ROOT/eval/simteb_audio_results"
python -u eval/analysis.py          | tail -5
python -u eval/make_tables.py       | tail -3
python -u eval/build_dashboard.py   | tail -2

echo
echo "finisher done $(date)"
date > logs/FINISHED
