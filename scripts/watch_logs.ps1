param(
    [ValidateSet("training", "dashboard", "all")]
    [string]$Target = "all",
    [int]$Tail = 80
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectRoot "logs"
$TrainingLog = Join-Path $LogDir "training.log"
$DashboardLog = Join-Path $LogDir "dashboard.log"

if ($Target -eq "training") {
    Get-Content -Path $TrainingLog -Tail $Tail -Wait
    exit
}

if ($Target -eq "dashboard") {
    Get-Content -Path $DashboardLog -Tail $Tail -Wait
    exit
}

Write-Host "Showing the latest $Tail lines from both logs. Use -Target training or -Target dashboard for live tail."
Write-Host ""
if (Test-Path $DashboardLog) {
    Write-Host "===== dashboard.log ====="
    Get-Content -Path $DashboardLog -Tail $Tail
}
if (Test-Path $TrainingLog) {
    Write-Host ""
    Write-Host "===== training.log ====="
    Get-Content -Path $TrainingLog -Tail $Tail
}
