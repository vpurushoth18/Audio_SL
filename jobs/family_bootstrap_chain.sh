#!/bin/bash
# Re-run the family-blocked bootstrap once the expansion finishes.
#
# Fires when EITHER
#   - 45 checkpoints are complete (the full target), or
#   - no expansion worker is alive any more (the run ended early, stopped, or a
#     checkpoint failed) -- in which case it still runs, on whatever finished,
#     and the log records the count it actually used.
#
# The second condition matters: waiting only on "45" would hang forever if a
# checkpoint fails, which is exactly what m-ctc-t-large did.
#
# Worker detection reads /proc and requires argv to be
#   bash <...>/jobs/expansion_worker{,5}.sh <gpu> ...
# skipping this script's own pid and its parent. The wave-4 chain died because
# `pgrep -f` matched the shell that WROTE it, the pattern being present in that
# shell's own cmdline.
set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1

TARGET=45
STOP="logs/STOP_EXPANSION"
OUT="logs/family_bootstrap_final.log"

n_complete() {
  python3 - <<'PY'
import pandas as pd
try:
    d = pd.read_csv("eval/analysis/expansion_status.tsv", sep="\t")
    print(int(((d.complete) & (~d.is_baseline)).sum()))
except Exception:
    print(0)
PY
}

workers_alive() {
  python3 - "$$" <<'PY'
import os, sys
skip = {int(sys.argv[1]), os.getppid()}
n = 0
for d in os.listdir("/proc"):
    if not d.isdigit() or int(d) in skip:
        continue
    try:
        with open(f"/proc/{d}/cmdline", "rb") as fh:
            parts = [p.decode("utf-8", "replace")
                     for p in fh.read().split(b"\0") if p]
    except OSError:
        continue
    if len(parts) < 3:
        continue
    if not parts[0].endswith("bash"):
        continue
    if not (parts[1].endswith("jobs/expansion_worker.sh")
            or parts[1].endswith("jobs/expansion_worker5.sh")):
        continue
    n += 1
print(n)
PY
}

echo "family-bootstrap chain started $(date); target $TARGET checkpoints"
while :; do
  N=$(n_complete)
  ALIVE=$(workers_alive)
  if [ -f "$STOP" ]; then
    echo "  [$(date +%T)] stop flag set; running on $N checkpoints"; break
  fi
  if [ "$N" -ge "$TARGET" ]; then
    echo "  [$(date +%T)] target reached: $N checkpoints"; break
  fi
  if [ "$ALIVE" -eq 0 ]; then
    echo "  [$(date +%T)] no workers alive; expansion ended at $N checkpoints"
    break
  fi
  echo "  [$(date +%T)] $N/$TARGET complete, $ALIVE worker(s) running"
  sleep 600
done

# let a checkpoint's own analysis refresh land first
sleep 60
N=$(n_complete)
echo; echo "running family-blocked bootstrap on $N checkpoints ($(date))"

# keep the expansion's own analysis refresh from overlapping this read
flock -w 1800 logs/.analysis.lock \
  env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
  nice -n 15 python3 -u eval/family_blocked_bootstrap.py > "$OUT" 2>&1
rc=$?

echo "bootstrap exit $rc ($(date)); output in $OUT"
if [ "$rc" -eq 0 ]; then
  date > logs/FAMILY_BOOTSTRAP_FINAL_DONE
  echo "--- headline ---"
  sed -n '/HEADLINE SEPARATION/,/SURVIVES/p' "$OUT"
fi
