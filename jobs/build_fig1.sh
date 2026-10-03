#!/bin/bash
# Wait for the layer sweep to write its table, then build the figures.
#
#   setsid nohup jobs/build_fig1.sh > logs/build_fig1.log 2>&1 < /dev/null &
#
# Detached, so it finishes whether or not the session that launched it is still
# alive. It waits on the OUTPUT FILE rather than on a process name: a pattern
# like `pgrep -f layer_curves` matches the waiting shell itself, because the
# pattern appears in that shell's own command line, and the loop then never
# exits. That mistake cost several hours earlier today.

set -uo pipefail
ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT" || exit 1

CURVES="eval/analysis/V_layer_curves.tsv"
DEADLINE=$(( $(date +%s) + 7200 ))        # give up after two hours

echo "waiting for $CURVES  (started $(date))"
while [ ! -f "$CURVES" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "!!! timed out after 2h; $CURVES never appeared"
    exit 1
  fi
  sleep 30
done

# the sweep writes the file in one go at the end, but settle anyway
sleep 5
echo "table appeared $(date): $(wc -l < "$CURVES") rows"

echo
echo ">>> building figures"
python -u eval/figures.py
rc=$?

echo
if [ -f eval/figures/fig1_layer_curves.pdf ]; then
  echo "fig1 written $(date)"
  ls -la eval/figures/
else
  echo "!!! fig1 missing after figures.py (exit $rc)"
fi
date > logs/FIG1_DONE
