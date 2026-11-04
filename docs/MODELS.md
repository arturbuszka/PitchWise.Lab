# Models and data

The pipeline runs ONNX models through ONNX Runtime. Nothing here needs Python at runtime.

## Get the models

```bash
tools/models/download.sh        # Windows: tools/models/download.ps1
```

This fetches about 750 MB into `pretrained_models/` from the
[`models-v1` release](https://github.com/arturbuszka/PitchWise.Lab/releases/tag/models-v1) and checks
each file against `tools/models/models.sha256`.

Only `yolo11m.onnx` is required. The other modules default to `enabled: auto` and turn on when their
files are present.

| Module | Files | Purpose | Source | License |
|---|---|---|---|---|
| Detection | `yolo11m.onnx` | People | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| ReID, role | `prtreid-soccernet-baseline.onnx` | Appearance and role | [PRTReid weights](https://zenodo.org/records/10653453) | CC-BY-4.0 |
| Calibration | `nbjw_kp.onnx`, `nbjw_lines.onnx` and their `.onnx.data` | Pitch homography | [NBJW weights](https://zenodo.org/records/12626395) | CC-BY-4.0 |
| Jersey OCR | `mmocr_det.onnx`, `mmocr_rec.onnx`, `mmocr_charset.txt` | Shirt numbers | [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 |

All files were converted to ONNX for this project; the weights are otherwise unchanged. The OCR files
keep their `mmocr_` names because they replace MMOCR from the original baseline, whose export
toolchain needs Python 3.11 or older and torch 2.1 or older.

Licenses above cover the published weights. The upstream code has its own terms: PRTReid is under
the Hippocratic License 3.0 and NBJW under GPL-2.0.

## Citations

- NBJW: Gutiérrez-Pérez, Agudo. *No Bells, Just Whistles: Sports Field Registration by Leveraging
  Geometric Properties.* CVPR Workshops 2024.
- PRTReid: Somers et al. *Multi-task Learning for Joint Re-identification, Team Affiliation, and Role
  Classification for Sports Visual Tracking.* MMSports 2023.

## Run on a GPU

The default config asks for CUDA and falls back to CPU, with a warning, when it is unavailable.
To stay on CPU without the warnings, pass `-c configs/cpu.yaml`.

ONNX Runtime needs CUDA 12 and cuDNN 9. If they are not installed system-wide, the NVIDIA wheels are
enough:

```bash
python3 -m venv ~/cuda12-libs
~/cuda12-libs/bin/pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 \
    "nvidia-cudnn-cu12~=9.10" nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-nvjitlink-cu12
export LD_LIBRARY_PATH="$(ls -d ~/cuda12-libs/lib/python3*/site-packages/nvidia/*/lib | paste -sd:)"
```

`tools/eval/run_eval.sh` sets this path itself. On GPUs with more than 6 GB, set
`onnx.low_vram: false` in the config.

## Regenerate the models

You only need this to change a model or to audit where the published ones came from.

```bash
python3.10 -m venv ~/.venvs/pitchwise-export
~/.venvs/pitchwise-export/bin/pip install -r tools/export/requirements.txt
tools/export/run_exports.sh
```

The script downloads the upstream weights and clones `sn-gamestate` into `reference/`, pinned to the
commit the models were exported with. The OCR step needs its own environment, described at the top
of `tools/export/export_ppocr.sh`. Exporting again may not reproduce the same bytes, so
`models.sha256` describes the published files only.

## Get the evaluation data

GS-HOTA needs the SoccerNet Game State Reconstruction dataset, which is not redistributed here:

```bash
pip install SoccerNet
python - <<'PY'
from SoccerNet.Downloader import SoccerNetDownloader
SoccerNetDownloader(LocalDirectory="data/SoccerNetGS").downloadDataTask(task="gamestate-2024", split=["valid"])
PY
unzip data/SoccerNetGS/gamestate-2024/valid.zip -d data/SoccerNetGS/valid
```

Then evaluate one sequence with
`dotnet run --project src/PitchWise.Cli -c Release -- eval data/SoccerNetGS valid SNGS-021`,
or the fixed benchmark set with `tools/eval/run_eval.sh`.
