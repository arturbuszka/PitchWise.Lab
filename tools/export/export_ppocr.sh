#!/usr/bin/env bash
# Converts the PP-OCRv4 detector and recogniser used for jersey numbers to ONNX.
# Needs its own venv, because paddle2onnx 1.3.1 does not fit the main export environment:
#   python3.10 -m venv ~/.venvs/pitchwise-ppocr
#   ~/.venvs/pitchwise-ppocr/bin/pip install paddle2onnx==1.3.1 paddlepaddle==2.6.2 "setuptools<81"
# Usage: PADDLE_VENV=~/.venvs/pitchwise-ppocr tools/export/export_ppocr.sh [models_dir]
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
models="$(mkdir -p "${1:-$repo/pretrained_models}" && cd "${1:-$repo/pretrained_models}" && pwd)"
p2o="${PADDLE_VENV:-$HOME/.venvs/pitchwise-ppocr}/bin/paddle2onnx"
[ -x "$p2o" ] || { echo "Missing $p2o (see the header of this script)"; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

fetch() { curl -fsSL --retry 5 --retry-all-errors -o "$2" "$1"; }
fetch https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar det.tar
fetch https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar rec.tar
fetch https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt en_dict.txt
tar xf det.tar
tar xf rec.tar

convert() {
    "$p2o" --model_dir "$1" --model_filename inference.pdmodel --params_filename inference.pdiparams \
        --save_file "$2" --opset_version 14 --enable_onnx_checker True
}
convert ch_PP-OCRv4_det_infer "$models/mmocr_det.onnx"
convert en_PP-OCRv4_rec_infer "$models/mmocr_rec.onnx"

# the recogniser's dictionary, plus the space class it appends
{ cat en_dict.txt; echo " "; } > "$models/mmocr_charset.txt"
echo "Exported PP-OCRv4 to $models"
