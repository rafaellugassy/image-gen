<#
.SYNOPSIS
  One-shot setup for FLUX.2 image generation on Windows (NVIDIA or AMD GPU) via ComfyUI.

.DESCRIPTION
  1. Checks for git, a supported Python and detects your GPU (NVIDIA or AMD)
  2. Clones ComfyUI into .\ComfyUI
  3. Creates a virtualenv in .\.venv and installs the matching PyTorch build
     (CUDA for NVIDIA, ROCm for AMD) + ComfyUI requirements
  4. Downloads the model files for the chosen profile

.EXAMPLE
  .\setup.ps1                       # auto-detect GPU, klein-4b (recommended for 16 GB VRAM)
  .\setup.ps1 -Gpu amd              # force AMD (ROCm) PyTorch
  .\setup.ps1 -Gpu amd -AmdDevice device-gfx1100   # smaller download, RX 7900 XT/XTX
  .\setup.ps1 -Profile klein-9b     # needs HF_TOKEN in .env
  .\setup.ps1 -Profile dev          # ~54 GB, needs lots of system RAM
  .\setup.ps1 -SkipModels           # install only, download models later
  .\setup.ps1 -CudaTag cu128        # NVIDIA only: if your driver is too old for cu130

  If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
param(
    [ValidateSet("klein-4b", "klein-4b-base", "klein-9b", "dev")]
    [string]$Profile = "klein-4b",
    [ValidateSet("auto", "nvidia", "amd")]
    [string]$Gpu = "auto",
    [ValidateSet("cu126", "cu128", "cu130")]
    [string]$CudaTag = "cu130",
    # AMD only. device-all installs kernels for every supported GPU (largest download).
    # Narrower options: device-gfx1201 (RX 9070), device-gfx1200 (RX 9060), device-gfx1100 (RX 7900),
    # device-gfx1101 (RX 7700/7800), device-gfx1102 (RX 7600), device-gfx1151 (Ryzen AI Max).
    [string]$AmdDevice = "device-all",
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

# --- GPU detection -----------------------------------------------------------
$gpuNames = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
Write-Host ("Detected display adapters: " + ($gpuNames -join "; "))
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue) -or [bool]($gpuNames -match "NVIDIA")
$hasAmd = [bool]($gpuNames -match "AMD|Radeon")

if ($Gpu -eq "auto") {
    if ($hasNvidia) { $Gpu = "nvidia" }
    elseif ($hasAmd) { $Gpu = "amd" }
    else { Fail "No NVIDIA or AMD GPU detected. Re-run with -Gpu nvidia or -Gpu amd to force one." }
}
Write-Host "Using GPU backend: $Gpu" -ForegroundColor Green

if ($Gpu -eq "nvidia" -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
if ($Gpu -eq "amd") {
    Write-Host "AMD on Windows needs Windows 11, a current AMD Adrenalin driver and an RDNA 2/3/3.5/4 GPU (RX 6000 series or newer)." -ForegroundColor Yellow
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

if ($Gpu -eq "amd") {
    $venvVer = (& $venvPy -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
    if ($venvVer -ne "3.13") {
        Fail "AMD ROCm PyTorch on Windows needs Python 3.13 but the venv uses $venvVer. Install Python 3.13 (64-bit), delete the .venv folder, and re-run."
    }
    # Versions pinned as in ComfyUI's README for AMD GPUs on Windows (ROCm 10.0, PyTorch 2.13).
    # If a previous NVIDIA install left a CUDA torch in the venv, remove it first.
    & $venvPy -m pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Null
    Step "Installing PyTorch for AMD (ROCm, $AmdDevice)"
    & $venvPy -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
        "torch[$AmdDevice]==2.13.0+rocm10.0.0" "torchvision[$AmdDevice]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
    Check "PyTorch (ROCm) install"
} else {
    Step "Installing PyTorch for NVIDIA ($CudaTag)"
    & $venvPy -m pip install torch torchvision torchaudio --extra-index-url "https://download.pytorch.org/whl/$CudaTag"
    Check "PyTorch install"
}

Step "Installing ComfyUI requirements"
& $venvPy -m pip install -r ComfyUI\requirements.txt
Check "ComfyUI requirements"
& $venvPy -m pip install -r requirements.txt
Check "project requirements"

Step "Verifying GPU access from PyTorch"
# ROCm builds of PyTorch also report through the torch.cuda API.
& $venvPy -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__); print('GPU available:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'none')"
Check "torch check"
$gpuOk = (& $venvPy -c "import torch; print(torch.cuda.is_available())").Trim()
if ($gpuOk -ne "True") {
    if ($Gpu -eq "amd") {
        Write-Host "WARNING: PyTorch cannot see your GPU. Update your AMD Adrenalin driver and confirm you are on Windows 11 with an RX 6000 series or newer GPU." -ForegroundColor Yellow
    } else {
        Write-Host "WARNING: PyTorch cannot see your GPU. Update your NVIDIA driver, or re-run with -CudaTag cu128 / cu126." -ForegroundColor Yellow
    }
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
