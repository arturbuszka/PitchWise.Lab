# Downloads the pre-exported ONNX models from the GitHub Release and verifies their checksums.
# About 750 MB. Files that already match are left alone, so it is safe to re-run.
# Usage: tools/models/download.ps1 [models_dir]
param([string]$Models = (Join-Path $PSScriptRoot "../../pretrained_models"))
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar slows Invoke-WebRequest badly
$base = if ($env:MODELS_URL) { $env:MODELS_URL } else { "https://github.com/arturbuszka/PitchWise.Lab/releases/download/models-v1" }
New-Item -ItemType Directory -Force -Path $Models | Out-Null

foreach ($line in Get-Content (Join-Path $PSScriptRoot "models.sha256")) {
    $sum, $name = $line -split '\s+', 2
    $target = Join-Path $Models $name
    if ((Test-Path $target) -and (Get-FileHash $target -Algorithm SHA256).Hash -ieq $sum) {
        Write-Host "ok        $name"
        continue
    }
    Write-Host "download  $name"
    Invoke-WebRequest "$base/$name" -OutFile "$target.part"
    if ((Get-FileHash "$target.part" -Algorithm SHA256).Hash -ine $sum) {
        Remove-Item "$target.part"
        throw "checksum mismatch: $name"
    }
    Move-Item -Force "$target.part" $target
}
Write-Host "Models ready in $Models"
