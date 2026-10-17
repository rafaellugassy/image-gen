<#
.SYNOPSIS
  Wrapper that runs generate.py inside the project's virtualenv.
.EXAMPLE
  .\generate.ps1 "a red fox in snow" --count 4 --width 1024 --height 768
#>
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run .\setup.ps1 first." -ForegroundColor Red; exit 1 }
& $py (Join-Path $PSScriptRoot "generate.py") @args
exit $LASTEXITCODE
