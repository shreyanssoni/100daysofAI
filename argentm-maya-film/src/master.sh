#!/usr/bin/env bash
# Render, then master the audio to -16 LUFS (two-pass loudnorm), video stream copied untouched.
# Usage: bash src/master.sh
set -euo pipefail
cd "$(dirname "$0")/.."
node src/build.mjs
python3 src/music.py
npx hyperframes render --quality looks --output renders/_raw.mp4
read -r I TP LRA TH OFF < <(ffmpeg -i renders/_raw.mp4 -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 \
  | sed -n '/{/,/}/p' | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['input_i'],d['input_tp'],d['input_lra'],d['input_thresh'],d['target_offset'])")
ffmpeg -y -loglevel error -i renders/_raw.mp4 -t 60 -c:v copy \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=$I:measured_TP=$TP:measured_LRA=$LRA:measured_thresh=$TH:offset=$OFF:linear=true" \
  -ar 48000 -c:a aac -b:a 256k -movflags +faststart renders/maya-60s-1080p.mp4
rm renders/_raw.mp4
