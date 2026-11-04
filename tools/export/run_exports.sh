#!/usr/bin/env bash
# Regenerates the ONNX models from the upstream weights (Zenodo, GitHub). Most people want
# tools/models/download.sh instead. See docs/MODELS.md.
# Usage: VENV=/path/to/venv ./run_exports.sh [models_dir]
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="${VENV:-$HOME/.venvs/pitchwise-export}"
models="${1:-$repo/pretrained_models}"
py="$venv/bin/python"
mkdir -p "$models"

[ -x "$py" ] || { echo "Missing venv at $venv (set VENV=..., requirements in tools/export/requirements.txt)"; exit 1; }

# export_nbjw.py imports the model code from this checkout, pinned to the commit used for the models
ref="$repo/reference/sn-gamestate"
if [ ! -d "$ref" ]; then
    git clone https://github.com/SoccerNet/sn-gamestate "$ref"
    git -C "$ref" checkout -q 1c958345067218297d221e45e1a6405f975f83e0
fi

echo "=== 1/4 YOLO (detection) ==="
(cd "$models" && "$py" "$repo/tools/export/export_yolo.py")

echo "=== 2/4 NBJW (calibration: lines + homography + minimap) ==="
"$py" "$repo/tools/export/export_nbjw.py" "$models"

echo "=== 3/4 PRTReid (teams + roles) ==="
"$py" "$repo/tools/export/export_prtreid.py" "$models"

echo "=== 4/4 PP-OCRv4 (jersey numbers) ==="
"$repo/tools/export/export_ppocr.sh" "$models"

echo "Done. Models in $models — the track command picks them up automatically."
