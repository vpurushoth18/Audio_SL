#!/bin/bash
# Wait for the two corrected reruns, then revalidate and repackage.
#
# Worker detection reads /proc and requires argv to be
#   bash <...>/jobs/rerun_corrected.sh <gpu> <model>
# skipping this script's own pid and its parent. `pgrep -f` is deliberately not
# used: earlier in this project it matched the shell that had written the script
# being waited on, because the pattern sat in that shell's own cmdline, and a
# chain never fired as a result.
set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1

SCRATCH="${1:?usage: audit_followup_chain.sh <scratchdir>}"
EXCL="$SCRATCH/exclude.txt"
SIN="$R/simteb_audio/mteb_cache_corrected"
MAEB="$R/simteb_audio/maeb_cache_corrected"
MODELS=("facebook/wav2vec2-conformer-rel-pos-large"
        "facebook/wav2vec2-xls-r-300m-21-to-en")

reruns_alive() {
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
    if len(parts) < 3 or not parts[0].endswith("bash"):
        continue
    if parts[1].endswith("jobs/rerun_corrected.sh"):
        n += 1
print(n)
PY
}

echo "audit follow-up chain started $(date)"
while :; do
  n=$(reruns_alive)
  [ "$n" -eq 0 ] && { echo "  [$(date +%T)] reruns finished"; break; }
  echo "  [$(date +%T)] $n rerun(s) still running"
  sleep 300
done

sleep 30

# ---- decide inclusion from actual cell counts, not from exit codes
: > "$EXCL"
echo "# Checkpoints excluded: corrected rerun did not produce a full task set." >> "$EXCL"
for m in "${MODELS[@]}"; do
  safe="${m//\//__}"
  ns=$(find "$SIN/results/$safe" -name '*.json' ! -name 'model_meta.json' 2>/dev/null | wc -l)
  nm=$(find "$MAEB/results/$safe" -name '*.json' ! -name 'model_meta.json' 2>/dev/null | wc -l)
  echo "  $m -> SinAEB $ns/14  MAEB $nm/19"
  if [ "$ns" -lt 14 ] || [ "$nm" -lt 19 ]; then
    echo "$m" >> "$EXCL"
    echo "    EXCLUDED (incomplete corrected rerun)"
  else
    echo "    validated, will be overlaid"
  fi
done

echo; echo "running revalidated analysis $(date)"
flock -w 1800 logs/.analysis.lock \
  env PYTHONPATH="$R/simteb_audio/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
  nice -n 15 python3 -u eval/revalidated_analysis.py --exclude-file "$EXCL" \
  > logs/revalidated_analysis.log 2>&1
rc=$?
echo "revalidated_analysis exit $rc"
[ "$rc" -eq 0 ] && date > logs/REVALIDATION_DONE
echo "chain finished $(date)"
