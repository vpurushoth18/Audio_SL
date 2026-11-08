#!/bin/bash
# Wave 5: the single substitute checkpoint, run on the first GPU to come free.
#
# The waves 1-4 workers built their model list at startup, so appending wave 5
# to the inventory does not reach them. This waits for one to exit and then
# claims its GPU.
#
# The wave-4 chain script failed because it waited on
#   pgrep -f "expansion_worker.sh [01] "
# which also matched the interactive shell that WROTE the script, since the
# heredoc text sat in that shell's own /proc cmdline. It therefore saw a worker
# running forever and never launched. The wait here reads /proc directly,
# requires the cmdline to BEGIN with the worker path, and skips its own pid and
# its parents, so it cannot match anything but a real worker.
set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1

MODEL="facebook/wav2vec2-conformer-rel-pos-large"
STOP="logs/STOP_EXPANSION"

# prints the gpu ids (0/1) that still have a live expansion worker
live_workers() {
  python3 - "$$" <<'PY'
import os, sys
self_pid = int(sys.argv[1])
skip = {self_pid, os.getppid()}
for d in os.listdir("/proc"):
    if not d.isdigit() or int(d) in skip:
        continue
    try:
        with open(f"/proc/{d}/cmdline", "rb") as fh:
            parts = fh.read().split(b"\0")
    except OSError:
        continue
    parts = [p.decode("utf-8", "replace") for p in parts if p]
    if len(parts) < 3:
        continue
    # argv must be exactly: bash <...>expansion_worker.sh <gpu> ...
    if not parts[0].endswith("bash"):
        continue
    if not parts[1].endswith("jobs/expansion_worker.sh"):
        continue
    print(parts[2])
PY
}

echo "wave 5 waiting for a free gpu ($(date))"
GPU=""
while :; do
  [ -f "$STOP" ] && { echo "stop flag set; wave 5 not starting"; exit 0; }
  mapfile -t BUSY < <(live_workers)
  for g in 0 1; do
    hit=0
    for b in "${BUSY[@]:-}"; do [ "$b" = "$g" ] && hit=1; done
    [ "$hit" -eq 0 ] && { GPU="$g"; break; }
  done
  [ -n "$GPU" ] && break
  echo "  [$(date +%T)] both gpus busy (workers on: ${BUSY[*]:-none})"
  sleep 300
done

echo; echo "gpu $GPU free at $(date); running $MODEL"

setsid nohup ./jobs/expansion_worker5.sh "$GPU" 0 1 5 \
  >> "logs/expansion_gpu${GPU}.log" 2>&1 &
echo "wave 5 worker launched on gpu $GPU ($(date))"
date > logs/WAVE5_LAUNCHED
