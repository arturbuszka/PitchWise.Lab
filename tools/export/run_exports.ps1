# Eksport wszystkich modeli baseline'u do ONNX (uruchom recznie — pobiera wagi
# z Zenodo/GitHub, czego agent nie moze zrobic bez Twojej zgody).
$ErrorActionPreference = "Stop"
$venv = "d:\tmp\sn-export\venv\Scripts"
$repo = "d:\repos\dotnet-sn-gamestate"
$models = "$repo\pretrained_models"

Write-Host "=== 1/3 NBJW (kalibracja: linie + homografia + minimapa) ==="
& "$venv\python.exe" "$repo\tools\export\export_nbjw.py" $models
if ($LASTEXITCODE -ne 0) { throw "export_nbjw failed ($LASTEXITCODE)" }

Write-Host "=== 2/3 PRTReid (druzyny + role) ==="
& "$venv\python.exe" "$repo\tools\export\export_prtreid.py" $models
if ($LASTEXITCODE -ne 0) { throw "export_prtreid failed ($LASTEXITCODE)" }

Write-Host "=== 3/3 MMOCR (numery koszulek) — opcjonalnie, najciezsze zaleznosci ==="
Write-Host "Instrukcja w $repo\tools\export\export_mmocr.py (wymaga mmdeploy + mmcv)."

Write-Host "Gotowe. Modele w $models — komenda track podlaczy je automatycznie."
