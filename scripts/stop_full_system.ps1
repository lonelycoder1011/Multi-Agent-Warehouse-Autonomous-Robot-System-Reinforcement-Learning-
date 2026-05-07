$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StackFile = Join-Path $ProjectRoot ".run\warehouse-stack.json"

if (-not (Test-Path $StackFile)) {
    Write-Host "No stack file found at $StackFile"
    exit 0
}

$stack = Get-Content $StackFile -Raw | ConvertFrom-Json
$pids = @($stack.dashboard.pid, $stack.training.pid) | Where-Object { $_ }

foreach ($pidValue in $pids) {
    try {
        taskkill /PID $pidValue /T /F | Out-Null
        Write-Host "Stopped process tree $pidValue"
    } catch {
        Write-Host "Process $pidValue was not running or could not be stopped."
    }
}

Remove-Item -Path $StackFile -Force
Write-Host "Warehouse RL stack stopped."
