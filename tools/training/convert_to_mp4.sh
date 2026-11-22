#!/usr/bin/env bash
# Convert iPhone recordings (HEVC .MOV) to H.264 .mp4 for Roboflow.
# Originals stay untouched; output goes to <dir of input>/mp4/<name>.mp4. Existing outputs are skipped.
#
#   tools/training/convert_to_mp4.sh ~/Studia/PROJEKTP/data_set_training/*.MOV
#   FPS=30 tools/training/convert_to_mp4.sh IN.MOV     # drop frames: Roboflow samples ~3/s anyway
#
# Encodes on the GPU (NVENC) when it actually works, otherwise on the CPU. ffmpeg listing
# h264_nvenc is not enough: with the nouveau driver there is no CUDA device and NVENC fails.
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: [FPS=30] $0 FILE.MOV [FILE.MOV ...]"; exit 1; }

if ffmpeg -v error -f lavfi -i color=size=256x256:duration=0.1 -c:v h264_nvenc -f null - 2>/dev/null; then
  codec=(-c:v h264_nvenc -preset p5 -rc vbr -cq 19 -b:v 0)
else
  codec=(-c:v libx264 -preset veryfast -crf 18)
fi
rate=()
[ -n "${FPS:-}" ] && rate=(-r "$FPS")

for in in "$@"; do
  out="$(dirname "$in")/mp4/$(basename "${in%.*}").mp4"
  if [ -s "$out" ]; then echo "skip (exists): $out"; continue; fi
  mkdir -p "$(dirname "$out")"
  echo "converting (${codec[1]}): $in -> $out"
  # Write to a temp name so a failed or interrupted run never leaves a file that looks done.
  tmp="${out%.mp4}.part.mp4"
  if nice ffmpeg -v error -stats -y -i "$in" -map 0:v:0 "${rate[@]}" "${codec[@]}" \
       -pix_fmt yuv420p -an -movflags +faststart "$tmp"; then
    mv "$tmp" "$out"
  else
    rm -f "$tmp"; echo "FAILED: $in" >&2
  fi
done
