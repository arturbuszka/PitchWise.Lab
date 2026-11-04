# Regenerates the ONNX models from the upstream weights (Zenodo, GitHub). Most people want
# tools/models/download.ps1 instead. See docs/MODELS.md.
# Usage: $env:VENV = "path\to\venv"; tools/export/run_exports.ps1 [models_dir]
# Covers YOLO, NBJW and PRTReid. The PP-OCRv4 conversion (export_ppocr.sh) is bash only.
param([string]$Models)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not $Models) { $Models = Join-Path $repo "pretrained_models" }
$venv = if ($env:VENV) { $env:VENV } else { Join-Path $HOME ".venvs/pitchwise-export" }
$py = if ($IsLinux -or $IsMacOS) { Join-Path $venv "bin/python" } else { Join-Path $venv "Scripts/python.exe" }
if (-not (Test-Path $py)) { throw "Missing venv at $venv (set VENV, requirements in tools/export/requirements.txt)" }
New-Item -ItemType Directory -Force -Path $Models | Out-Null

# export_nbjw.py imports the model code from this checkout, pinned to the commit used for the models
$ref = Join-Path $repo "reference/sn-gamestate"
if (-not (Test-Path $ref)) {
    git clone https://github.com/SoccerNet/sn-gamestate $ref
    git -C $ref checkout -q 1c958345067218297d221e45e1a6405f975f83e0
}

Write-Host "=== 1/3 YOLO (detection) ==="
Push-Location $Models
try { & $py (Join-Path $repo "tools/export/export_yolo.py") } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "export_yolo failed ($LASTEXITCODE)" }

Write-Host "=== 2/3 NBJW (calibration) ==="
& $py (Join-Path $repo "tools/export/export_nbjw.py") $Models
if ($LASTEXITCODE -ne 0) { throw "export_nbjw failed ($LASTEXITCODE)" }

Write-Host "=== 3/3 PRTReid (teams + roles) ==="
& $py (Join-Path $repo "tools/export/export_prtreid.py") $Models
if ($LASTEXITCODE -ne 0) { throw "export_prtreid failed ($LASTEXITCODE)" }

Write-Host "Done. Models in $Models"
