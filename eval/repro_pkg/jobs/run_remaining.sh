#!/bin/bash
# Everything still outstanding, run sequentially and detached.
#
#   setsid nohup jobs/run_remaining.sh > logs/remaining.log 2>&1 < /dev/null &
#
# Safe to leave unattended: each step writes its own TSVs, a failure is logged
# and the next step still runs, and re-running skips work already cached.

set -uo pipefail

ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/simteb_audio/src"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

step () {
  echo
  echo "=================================================================="
  echo ">>> $1   ($(date +%T))"
  echo "=================================================================="
  shift
  "$@"
  rc=$?
  [ "$rc" -ne 0 ] && echo "!!! FAILED (exit $rc) - continuing"
  return 0
}

echo "started $(date) on $(hostname), pid $$"

# 1. wait for the layer-transfer pass already running, so we do not run two
#    k-means sweeps over the same embeddings at once
while pgrep -f "eval/layer_transfer.py" >/dev/null; do sleep 30; done

# 2. controlled matching over all six models, final layer + full sweep.
#    wavlm and hubert were extracted after the first pass, so this supersedes it.
step "controlled matching, 6 models, final layer" \
  python -u eval/controlled_matching.py
step "controlled matching, 6 models, all layers" \
  python -u eval/controlled_matching.py --layers all

# 3. layer transfer with all six models: pick on dev, freeze, test held-out
step "layer transfer, 6 models" \
  python -u eval/layer_transfer.py

# 4. refresh the aggregate analyses now that the MFCC baseline is a row
step "ranking and capability analyses" \
  python -u eval/analysis.py

# 5. rebuild the result tables and the dashboard
step "collect results" \
  python -u eval/collect_mteb_results.py \
      --cache "$ROOT/simteb_audio/mteb_cache" \
      --out "$ROOT/eval/simteb_audio_results"
step "dashboard" python -u eval/build_dashboard.py

echo
echo "finished $(date)"
echo "outputs: eval/analysis/*.tsv  eval/audit/*.tsv  eval/dashboard.html"
