#!/usr/bin/env bash
# Wait out any running unpacker, then keep unpacking until all 16 parts are done
# and the downloader has exited; finally build the manifest.
set -u
R=/nfs/cc-filer/home/pvelayuthan/Documents/audio_SL
cd "$R" || exit 1
while pgrep -f unpack_openslr52.sh > /dev/null; do sleep 30; done
while :; do
  done_n=$(ls data/openslr52/extracted/.done_* 2>/dev/null | wc -l)
  dl_running=$(pgrep -f fetch_openslr52.sh > /dev/null && echo yes || echo no)
  echo "[$(date +%T)] extracted parts=$done_n/16 downloader=$dl_running"
  [ "$done_n" -eq 16 ] && break
  [ "$dl_running" = "no" ] && [ "$done_n" -eq "$(ls data/openslr52/*.zip 2>/dev/null | wc -l)" ] && break
  ./unpack_openslr52.sh
  sleep 10
done
echo "[$(date +%T)] extraction settled; building manifest"
python3 build_manifest.py
