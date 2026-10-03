#!/bin/bash
# Rebuild the figures once the layer sweep REPLACES its table.
#
# The table already exists from an earlier run, so waiting on the file's
# existence would fire immediately on stale data. This waits for its
# modification time to advance past the moment the watcher started, which is
# the only signal that the new sweep has written.

set -uo pipefail
ROOT=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$ROOT" || exit 1

CURVES="eval/analysis/V_layer_curves.tsv"
START=$(date +%s)
DEADLINE=$(( START + 5400 ))
OLD=$(stat -c %Y "$CURVES" 2>/dev/null || echo 0)
echo "watcher started $(date); current table mtime $OLD"

while :; do
  NOW=$(stat -c %Y "$CURVES" 2>/dev/null || echo 0)
  [ "$NOW" -gt "$OLD" ] && break
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "!!! timed out; table never refreshed"; exit 1
  fi
  sleep 30
done

sleep 5
echo "table refreshed $(date): $(wc -l < "$CURVES") rows"
echo
python -u eval/figures.py
echo
echo "done $(date)"
date > logs/FIG1_REBUILT
