#!/usr/bin/env bash
# Extract only OpenSLR-52 zips whose on-disk size matches the server's advertised length.
set -u
D=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL/data/openslr52
OUT=$D/extracted
mkdir -p "$OUT"
declare -A SZ=( [0]=915237858 [1]=908852134 [2]=913568157 [3]=901325452 [4]=922493671
  [5]=922505332 [6]=914729823 [7]=911992962 [8]=924344925 [9]=920427318 [a]=901532849
  [b]=924132571 [c]=938991415 [d]=911368918 [e]=927771260 [f]=917209429 )
for f in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  z=$D/asr_sinhala_$f.zip
  [ -f "$z" ] || { echo "[skip] $f not downloaded"; continue; }
  have=$(stat -c%s "$z")
  if [ "$have" != "${SZ[$f]}" ]; then
    echo "[skip] $f incomplete ($have/${SZ[$f]})"; continue
  fi
  if [ -f "$OUT/.done_$f" ]; then echo "[skip] $f already extracted"; continue; fi
  echo "[$(date +%T)] extracting $f"
  if unzip -qq -o "$z" -d "$OUT"; then touch "$OUT/.done_$f"; else echo "FAILED extract $f"; fi
done
echo "[$(date +%T)] flac count: $(find "$OUT" -name '*.flac' | wc -l)"
