param(
    [int]$DashboardPort = 8080,
    [string]$HostAddress = "127.0.0.1",
    [string]$TrainConfig = "configs/mappo_config.yaml",
    [string]$EnvConfig = "configs/env_config.yaml",
    [string]$CurriculumConfig = "configs/curriculum_config.yaml",
    [int]$NumWorkers = 2,
    [switch]$Resume,
    [string]$Checkpoint = "",
    [switch]$OnlineWandb
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot "rl-env\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @(
    "scripts\run_full_system.py",
    "--dashboard-port", "$DashboardPort",
    "--host", "$HostAddress",
    "--train-config", "$TrainConfig",
    "--env-config", "$EnvConfig",
    "--curriculum-config", "$CurriculumConfig",
    "--num-workers", "$NumWorkers"
)

if ($Resume) {
    $ArgsList += "--resume"
}
if ($Checkpoint) {
    $ArgsList += @("--checkpoint", "$Checkpoint")
}
if ($OnlineWandb) {
    $ArgsList += "--online-wandb"
}

& $Python @ArgsList
