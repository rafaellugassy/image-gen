<#
.SYNOPSIS
  Starts the FLUX Studio web UI server.

.DESCRIPTION
  Launches the lightweight, phone-friendly image generation web server.
  ComfyUI backend is automatically connected if running, or auto-started on demand.

.PARAMETER Port
  Port to listen on (default: 8080).

.PARAMETER Bind
  IP address to bind to (default: 127.0.0.1).

.PARAMETER Comfy
  ComfyUI host:port endpoint (default: 127.0.0.1:8188).

.PARAMETER Open
  Automatically opens the web UI in your default web browser upon launch.

.EXAMPLE
  .\run_server.ps1
  .\run_server.ps1 -Open
  .\run_server.ps1 -Port 8080 -Open
#>
[CmdletBinding()]
param(
    [int]$Port = 8080,
    [string]$Bind = "127.0.0.1",
    [string]$Comfy = "127.0.0.1:8188",
    [switch]$Open
)

# Use UTF-8 output encoding where supported for clean symbols
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Helper: Test TCP port connection quickly without hanging
function Test-TcpEndpoint([string]$HostName, [int]$PortNum, [int]$TimeoutMs = 400) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $asyncResult = $client.BeginConnect($HostName, $PortNum, $null, $null)
        $success = $asyncResult.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($success -and $client.Connected) {
            $client.EndConnect($asyncResult)
            return $true
        }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
        $client.Dispose()
    }
}

# Helper: Check if local port is currently in use
function Test-LocalPortInUse([int]$PortNum) {
    try {
        $properties = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties()
        $listeners = $properties.GetActiveTcpListeners()
        foreach ($endpoint in $listeners) {
            if ($endpoint.Port -eq $PortNum) { return $true }
        }
    } catch {}
    return $false
}

Clear-Host -ErrorAction SilentlyContinue

# --- Banner UI ---
Write-Host ""
Write-Host "  ================================================================" -ForegroundColor Cyan
Write-Host "   ✦  FLUX STUDIO  |  IMAGE GENERATION SERVER" -ForegroundColor White
Write-Host "      Fast, phone-friendly generative AI on your GPU" -ForegroundColor DarkGray
Write-Host "  ================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Environment Check
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "  [!] Virtual environment (.venv) not found!" -ForegroundColor Red
    Write-Host "      Please run the setup script first:" -ForegroundColor Yellow
    Write-Host "        .\setup.ps1" -ForegroundColor White
    Write-Host ""
    exit 1
}

# 2. Port check
if (Test-LocalPortInUse $Port) {
    Write-Host "  [!] Notice: Port $Port already appears to be in use." -ForegroundColor Yellow
    Write-Host "      If launch fails, specify another port: .\run_server.ps1 -Port <number>`n" -ForegroundColor DarkGray
}

# 3. ComfyUI Status Check
$comfyParts = $Comfy -split ":"
$comfyHost = if ($comfyParts.Length -ge 1) { $comfyParts[0] } else { "127.0.0.1" }
$comfyPort = if ($comfyParts.Length -ge 2) { [int]$comfyParts[1] } else { 8188 }
$comfyIsUp = Test-TcpEndpoint $comfyHost $comfyPort

# 4. Display Status Card
$localUrl = if ($Bind -eq "127.0.0.1" -or $Bind -eq "0.0.0.0") { "http://localhost:$Port" } else { "http://${Bind}:$Port" }

Write-Host "  STATUS & CONFIGURATION:" -ForegroundColor Cyan
Write-Host "    Web Address : " -NoNewline -ForegroundColor Gray
Write-Host "$localUrl" -ForegroundColor Green
Write-Host "    Bind Host   : " -NoNewline -ForegroundColor Gray
Write-Host "$Bind (Port $Port)" -ForegroundColor White
Write-Host "    ComfyUI     : " -NoNewline -ForegroundColor Gray

if ($comfyIsUp) {
    Write-Host "Connected ($Comfy) [Fast Mode]" -ForegroundColor Green
} else {
    Write-Host "On-Demand ($Comfy) [Starts automatically when generating]" -ForegroundColor Yellow
}

Write-Host "    Outputs     : " -NoNewline -ForegroundColor Gray
Write-Host ".\outputs\" -ForegroundColor DarkGray

Write-Host ""
Write-Host "  ----------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  PHONE & REMOTE ACCESS (Tailscale):" -ForegroundColor Cyan
Write-Host "    To use from your phone, run in a separate terminal:" -ForegroundColor Gray
Write-Host "      tailscale serve --bg $Port" -ForegroundColor Yellow
Write-Host "    Then open the secure https://... URL printed by Tailscale." -ForegroundColor Gray
Write-Host "  ----------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  [Ctrl + C] Stop Server" -ForegroundColor DarkGray
Write-Host ""

# Optional: Auto-open browser
if ($Open) {
    Write-Host "  Opening default browser to $localUrl..." -ForegroundColor Cyan
    Start-Process $localUrl
}

# Run the Python server
try {
    & $py (Join-Path $PSScriptRoot "server.py") --port $Port --bind $Bind --comfy $Comfy @args
} finally {
    Write-Host "`n  [Server Stopped] FLUX Studio server shut down." -ForegroundColor Gray
}
