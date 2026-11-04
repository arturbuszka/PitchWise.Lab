#!/usr/bin/env bash
# Downloads the pre-exported ONNX models from the GitHub Release and verifies their checksums.
# About 750 MB. Files that already match are left alone, so it is safe to re-run.
# Usage: tools/models/download.sh [models_dir]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models="${1:-$here/../../pretrained_models}"
base="${MODELS_URL:-https://github.com/arturbuszka/PitchWise.Lab/releases/download/models-v1}"
mkdir -p "$models"

while read -r sum name; do
    target="$models/$name"
    if [ -f "$target" ] && echo "$sum  $target" | sha256sum -c --status; then
        echo "ok        $name"
        continue
    fi
    echo "download  $name"
    curl -fL --retry 3 --progress-bar -o "$target.part" "$base/$name"
    echo "$sum  $target.part" | sha256sum -c --status || { echo "checksum mismatch: $name" >&2; rm -f "$target.part"; exit 1; }
    mv "$target.part" "$target"
done < "$here/models.sha256"

echo "Models ready in $models"
