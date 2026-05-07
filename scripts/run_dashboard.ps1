param(
    [int]$Port = 8080,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot "rl-env\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:WAREHOUSE_RL_DISABLE_RAY_ENV = "1"

Write-Host "Starting Warehouse RL dashboard..."
Write-Host "URL: http://$HostAddress`:$Port"
Write-Host "Press Ctrl+C to stop."

& $Python -m uvicorn dashboard.app:app --host $HostAddress --port $Port --reload
