#!/bin/bash
# Build the reproducibility package: code, metadata, configs, logs, results.
# Deliberately excludes audio and embedding caches, which stay on the cluster.
set -uo pipefail
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1
OUT="$R/eval/repro_pkg"
rm -rf "$OUT"; mkdir -p "$OUT"/{code,results,audit,tasks,jobs,env,splits,raw_results}

# --- experiment code
cp eval/*.py "$OUT/code/" 2>/dev/null
# --- task + model implementations, MAEB runner
mkdir -p "$OUT/tasks/simteb_audio"
cp -r simteb_audio/src/simteb_audio/* "$OUT/tasks/simteb_audio/" 2>/dev/null
find "$OUT/tasks" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
cp simteb_audio/maeb_evaluate.py "$OUT/tasks/" 2>/dev/null
# --- launch scripts
cp jobs/*.sh "$OUT/jobs/" 2>/dev/null
# --- result tables
cp eval/analysis/*.tsv "$OUT/results/" 2>/dev/null
cp eval/simteb_audio_results*.tsv "$OUT/results/" 2>/dev/null
cp eval/audit/*.tsv eval/audit/*.md "$OUT/audit/" 2>/dev/null
# --- split metadata (ids only, no audio)
cp eval/subset.tsv eval/subset_omni.tsv "$OUT/splits/" 2>/dev/null
# --- ORIGINAL evaluator outputs, so summaries can be checked against source
mkdir -p "$OUT/raw_results/sinhala" "$OUT/raw_results/maeb"
cp -r simteb_audio/mteb_cache/results/* "$OUT/raw_results/sinhala/" 2>/dev/null
cp -r simteb_audio/maeb_cache/results/* "$OUT/raw_results/maeb/" 2>/dev/null
# --- environment
{
  echo "# environment"; echo
  echo "## git"; git rev-parse HEAD 2>/dev/null; git status --porcelain | head -20
  echo; echo "## python"; python -V
  echo; echo "## key packages"
  pip list 2>/dev/null | grep -iE "^(mteb|torch|transformers|datasets|scikit-learn|numpy|scipy|pandas|soundfile|soxr|accelerate|huggingface)" 
  echo; echo "## hardware"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
  echo; echo "## host"; hostname; uname -r
} > "$OUT/env/environment.txt" 2>&1
pip freeze > "$OUT/env/pip_freeze.txt" 2>/dev/null
# --- run logs (trimmed: drop progress bars)
mkdir -p "$OUT/env/logs"
for f in logs/*.log; do
  [ -f "$f" ] || continue
  grep -avE "it/s\]|s/it\]|utt/s|examples/s" "$f" 2>/dev/null | tail -400 \
    > "$OUT/env/logs/$(basename "$f")"
done

echo "package contents:"
du -sh "$OUT"/* | sed 's/^/  /'
echo
echo "total: $(du -sh "$OUT" | cut -f1)"
