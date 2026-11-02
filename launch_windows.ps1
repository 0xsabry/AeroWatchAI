# AeroWatch AI — Windows launcher (Phase 10)
#
#   powershell -ExecutionPolicy Bypass -File launch_windows.ps1
#
# Runs preflight, then starts the app at http://localhost:8501.
# Stop with Ctrl+C (the app and history database shut down cleanly;
# history is stored in runs\history and survives restarts).
# Add -Offline to run with outbound network access blocked
# (AEROWATCH_OFFLINE=1) — the app needs no network at runtime.
#
#   ... launch_windows.ps1 -Offline

param([switch]$Offline)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[LAUNCH FAILED] No virtual environment found. Run setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

& $venvPython preflight.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[LAUNCH FAILED] Preflight failed — fix the [FAIL] items above first." -ForegroundColor Red
    exit 1
}

if ($Offline) {
    $env:AEROWATCH_OFFLINE = "1"
    Write-Host "Offline mode ON: outbound network connections will be blocked." -ForegroundColor Yellow
}

Write-Host "`nStarting AeroWatch AI at http://localhost:8501  (Ctrl+C to stop)" -ForegroundColor Cyan
& $venvPython -m streamlit run app.py
