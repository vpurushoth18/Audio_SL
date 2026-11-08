#!/bin/bash
# Last step of the reproducibility pass: wait for the per-seed table, refresh
# the figures that read it, and rebuild the collected package.
#
# Waits on the OUTPUT FILE, never on a process name: a `pgrep -f <script>` loop
# matches the waiting shell itself and never exits.
set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1
Q2="eval/analysis/Q2_per_seed_scores.tsv"
DEADLINE=$(( $(date +%s) + 5400 ))
rm -f logs/REPRO_DONE

echo "waiting for $Q2 ($(date))"
while [ ! -f "$Q2" ]; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "!!! timed out"; exit 1; }
  sleep 30
done
sleep 5
echo "per-seed table written: $(wc -l < "$Q2") rows"

echo; echo ">>> refreshing figures that read Q (fig5)"
python -u eval/figures.py 2>&1 | grep -vE "Warning|warn"

echo; echo ">>> rebuilding the collected package"
bash jobs/collect_repro.sh

echo; echo "finished $(date)"
date > logs/REPRO_DONE
