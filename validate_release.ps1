# AeroWatch AI — unified release validation (Phase 11)
#
#   powershell -ExecutionPolicy Bypass -File validate_release.ps1            # quick
#   powershell -ExecutionPolicy Bypass -File validate_release.ps1 -Full      # + GPU/CPU benchmark
#   powershell -ExecutionPolicy Bypass -File validate_release.ps1 -Camera    # + physical camera check
#   powershell -ExecutionPolicy Bypass -File validate_release.ps1 -Offline   # + offline-mode check
#   powershell -ExecutionPolicy Bypass -File validate_release.ps1 -Full -Camera -Offline
#
# Orchestrates preflight, hardware diagnostics, dependency checks,
# checkpoint integrity, ONNX smoke, database integrity and app
# startup via release_validation.py, and writes
# release_validation\report.json + report.md with explicit
# PASS / FAIL / NOT VERIFIED statuses. Hardware that is absent or
# not requested is reported NOT VERIFIED — never as a pass.

param(
    [switch]$Full,
    [switch]$Camera,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[VALIDATION FAILED] No virtual environment found. Run setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

$validationArgs = @("release_validation.py")
if ($Full)    { $validationArgs += "--full" }
if ($Camera)  { $validationArgs += "--camera" }
if ($Offline) { $validationArgs += "--offline" }

& $venvPython @validationArgs
exit $LASTEXITCODE
