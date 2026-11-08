#!/bin/bash
# Wave 4, launched after the waves 1-3 workers exit.
#
# The running workers built their model list at startup, so adding wave 4 to
# the inventory did not reach them. This waits for both to finish, then runs
# wave 4 split across the same two GPUs.
#
# It inherits the same stop behaviour: touch logs/STOP_EXPANSION and it will
# not start, or will stop after the checkpoint in flight.
set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1

echo "wave 4 waiting for waves 1-3 to finish ($(date))"
while pgrep -f "expansion_worker.sh [01] " >/dev/null 2>&1; do
  n=$(cat logs/expansion_done_gpu*.txt 2>/dev/null | wc -l)
  echo "  [$(date +%T)] waves 1-3 still running, $n/26 checkpoints complete"
  sleep 600
done

if [ -f logs/STOP_EXPANSION ]; then
  echo "stop flag set; wave 4 not starting"; exit 0
fi

echo; echo "waves 1-3 finished $(date). starting wave 4."
setsid nohup ./jobs/expansion_worker.sh 0 0 2 4 >> logs/expansion_gpu0.log 2>&1 &
sleep 5
setsid nohup ./jobs/expansion_worker.sh 1 1 2 4 >> logs/expansion_gpu1.log 2>&1 &
echo "wave 4 workers launched $(date)"
date > logs/WAVE4_LAUNCHED
