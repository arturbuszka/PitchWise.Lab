#!/usr/bin/env bash
# Export all baseline models to ONNX (run manually — downloads weights from
# Zenodo/GitHub). Usage: VENV=/path/to/venv ./run_exports.sh [models_dir]
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="${VENV:-$HOME/.venvs/pitchwise-export}"
models="${1:-$repo/pretrained_models}"
py="$venv/bin/python"
mkdir -p "$models"

[ -x "$py" ] || { echo "Missing venv at $venv (set VENV=...)"; exit 1; }

echo "=== 1/4 YOLO (detection) ==="
(cd "$models" && "$py" "$repo/tools/export/export_yolo.py")

echo "=== 2/4 NBJW (calibration: lines + homography + minimap) ==="
"$py" "$repo/tools/export/export_nbjw.py" "$models"

echo "=== 3/4 PRTReid (teams + roles) ==="
"$py" "$repo/tools/export/export_prtreid.py" "$models"

echo "=== 4/4 MMOCR (jersey numbers) — optional, heaviest dependencies ==="
echo "See $repo/tools/export/export_mmocr.py (needs mmdeploy + mmcv)."

echo "Done. Models in $models — the track command picks them up automatically."
