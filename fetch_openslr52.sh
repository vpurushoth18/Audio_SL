#!/usr/bin/env bash
# OpenSLR-52: Sinhala ASR, 16 zip parts + transcript TSV (~14.6 GB)
set -u
D=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL/data/openslr52
cd "$D" || exit 1
B=https://www.openslr.org/resources/52
wget -c -q --tries=5 --timeout=60 "$B/utt_spk_text.tsv"
for f in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  echo "[$(date +%T)] asr_sinhala_$f.zip"
  wget -c -q --tries=5 --timeout=60 "$B/asr_sinhala_$f.zip" \
    || echo "FAILED asr_sinhala_$f.zip"
done
echo "[$(date +%T)] done; $(du -sh "$D" | cut -f1)"
