<#
.SYNOPSIS
  Start the phone-friendly web page at http://127.0.0.1:8080
  (ComfyUI must be running: .\run_comfyui.ps1)
  For private access from your phone: tailscale serve --bg 8080
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run .\setup.ps1 first." -ForegroundColor Red; exit 1 }

& $py server.py @args
