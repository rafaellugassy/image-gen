<#
.SYNOPSIS
  Start the ComfyUI server/web UI at http://127.0.0.1:8188
  Keeping it running makes repeated generate.py calls faster (models stay loaded).
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run .\setup.ps1 first." -ForegroundColor Red; exit 1 }

Set-Location ComfyUI
& $py main.py --listen 127.0.0.1 --port 8188 @args
