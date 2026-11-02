# AeroWatch AI — Windows setup (Phase 10)
#
# One-time installation on Windows 10/11:
#   1. Open PowerShell in the project folder.
#   2. Run:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#
# What it does: checks Python >= 3.10, creates .venv, installs the
# pinned dependencies (CPU build by default), then runs preflight.py.
# It downloads NOTHING except the pip packages, installs NO GPU
# packages blindly, and never downloads model weights — if the v7
# checkpoint is missing, preflight tells you exactly what is needed.
#
# GPU (RTX 3050): the CPU install works everywhere. For CUDA
# acceleration, after this setup completes, run the printed GPU
# command to swap in the CUDA PyTorch build.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Fail($msg) {
    Write-Host "`n[SETUP FAILED] $msg" -ForegroundColor Red
    exit 1
}

Write-Host "=== AeroWatch AI — Windows setup ===" -ForegroundColor Cyan

# --- Python ------------------------------------------------------------
$python = $null
foreach ($candidate in @("py", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $python = $candidate
        break
    }
}
if (-not $python) {
    Fail "Python not found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run."
}
$version = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python: $version (via '$python')"
$parts = $version.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Fail "Python >= 3.10 is required (3.12 verified). Found $version."
}

# --- Virtual environment ----------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    & $python -m venv .venv
}
$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Fail "Virtual environment looks broken (no $venvPython). Delete the .venv folder and re-run."
}

# --- Dependencies (internet required for this step only) ---------------
Write-Host "Installing pinned dependencies (CPU build)..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Fail "pip install failed. Check your internet connection and re-run — the install is safe to repeat."
}

# --- GPU note (detect, never install blindly) ---------------------------
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    Write-Host "`nNVIDIA GPU detected." -ForegroundColor Green
    Write-Host "Optional CUDA acceleration — the CPU build already installed works,"
    Write-Host "but for RTX 3050 GPU inference run this inside the project folder:"
    Write-Host "  .venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Yellow
    Write-Host "Then verify with: .venv\Scripts\python.exe evaluation\hardware_diagnostics.py"
}

# --- Preflight -----------------------------------------------------------
Write-Host "`nRunning preflight checks..."
& $venvPython preflight.py
if ($LASTEXITCODE -ne 0) {
    Fail "Preflight reported failures — see the [FAIL] lines above (most commonly: model weights not yet copied into weights\)."
}
Write-Host "`nSetup complete. Launch with:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File launch_windows.ps1"
