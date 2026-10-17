<#
.SYNOPSIS
  One-shot setup for FLUX.2 image generation on Windows (NVIDIA GPU) via ComfyUI.

.DESCRIPTION
  1. Checks for git, NVIDIA GPU and a supported Python (3.10-3.13)
  2. Clones ComfyUI into .\ComfyUI
  3. Creates a virtualenv in .\.venv and installs PyTorch (CUDA) + ComfyUI requirements
  4. Downloads the model files for the chosen profile

.EXAMPLE
  .\setup.ps1                       # klein-4b (recommended for 16 GB VRAM)
  .\setup.ps1 -Profile klein-9b     # needs $env:HF_TOKEN
  .\setup.ps1 -Profile dev          # ~54 GB, needs lots of system RAM
  .\setup.ps1 -SkipModels           # install only, download models later
  .\setup.ps1 -CudaTag cu128        # if your NVIDIA driver is too old for cu130

  If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
param(
    [ValidateSet("klein-4b", "klein-4b-base", "klein-9b", "dev")]
    [string]$Profile = "klein-4b",
    [ValidateSet("cu126", "cu128", "cu130")]
    [string]$CudaTag = "cu130",
    [switch]$SkipModels
)

# "Continue" (not "Stop"): git/pip write progress to stderr, which Windows PowerShell 5.1
# would otherwise treat as fatal. Failures are caught explicitly via Check().
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)" } }

# --- prerequisites -----------------------------------------------------------
Step "Checking prerequisites"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git not found. Install it from https://git-scm.com/download/win and re-run."
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Write-Host "WARNING: nvidia-smi not found. This setup targets NVIDIA GPUs; install the latest NVIDIA driver." -ForegroundColor Yellow
}

# Find a Python between 3.10 and 3.13 (3.13 preferred by ComfyUI).
$pyExe = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
        $null = & py "-$v" -c "import sys" 2>&1
        if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-$v"); break }
    }
}
if (-not $pyExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $ver = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
    if ($LASTEXITCODE -eq 0 -and $ver -and ($ver.Trim() -in @("3.10", "3.11", "3.12", "3.13"))) { $pyExe = "python" }
}
if (-not $pyExe) {
    Fail "Python 3.10-3.13 (64-bit) not found. Install 3.13 from https://www.python.org/downloads/windows/ (tick 'Add to PATH') and re-run."
}
Write-Host "Using Python: $pyExe $($pyArgs -join ' ')"

# --- ComfyUI -----------------------------------------------------------------
Step "Getting ComfyUI"
if (Test-Path "ComfyUI\main.py") {
    Write-Host "ComfyUI already cloned; pulling latest"
    git -C ComfyUI pull --ff-only
    Check "git pull"
} else {
    git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
    Check "git clone"
}

# --- virtualenv --------------------------------------------------------------
Step "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pyExe @pyArgs -m venv .venv
    Check "venv creation"
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $venvPy -m pip install --upgrade pip
Check "pip upgrade"

Step "Installing PyTorch ($CudaTag)"
& $venvPy -m pip install torch torchvision torchaudio --extra-index-url "https://download.pytorch.org/whl/$CudaTag"
Check "PyTorch install"

Step "Installing ComfyUI requirements"
& $venvPy -m pip install -r ComfyUI\requirements.txt
Check "ComfyUI requirements"
& $venvPy -m pip install -r requirements.txt
Check "project requirements"

Step "Verifying GPU access from PyTorch"
& $venvPy -c "import torch; ok=torch.cuda.is_available(); print('CUDA available:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'none')"
Check "torch check"
$cuda = (& $venvPy -c "import torch; print(torch.cuda.is_available())").Trim()
if ($cuda -ne "True") {
    Write-Host "WARNING: PyTorch cannot see your GPU. Update your NVIDIA driver, or re-run with -CudaTag cu128 / cu126." -ForegroundColor Yellow
}

# --- models ------------------------------------------------------------------
if ($SkipModels) {
    Write-Host "`nSkipping model download. Later run:  .\.venv\Scripts\python download_models.py --profile $Profile"
} else {
    Step "Downloading models for profile '$Profile' (resumable - re-run if interrupted)"
    & $venvPy download_models.py --profile $Profile
    Check "model download"
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Generate an image:"
Write-Host "  .\generate.ps1 `"a lighthouse at dusk, cinematic`"" 
Write-Host "Or open the ComfyUI web UI:"
Write-Host "  .\run_comfyui.ps1"
