<#
.SYNOPSIS
  Start the ComfyUI server/web UI at http://127.0.0.1:8188
  Keeping it running makes repeated generate.py calls faster (models stay loaded).
.PARAMETER Profile
  Model profile to pre-load in the web UI (default: klein-4b).
  Run 'python download_models.py --list' to see all profiles.
#>
param([string]$Profile = "klein-4b")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run .\setup.ps1 first." -ForegroundColor Red; exit 1 }

# Write the chosen profile's model names into ComfyUI's default workflow so the
# web UI opens with the right models already wired up.
& $py (Join-Path $PSScriptRoot "write_default_workflow.py") --profile $Profile

Set-Location ComfyUI
& $py main.py --listen 127.0.0.1 --port 8188 @args
