# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

# =============================================================================
# run_profiles.ps1
# =============================================================================
# Runs each profile in turn with a cooldown before it, and saves each run to its own result folder
# under profiler\Results\ (override with -ResultsDir <dir>).
# PowerShell version of profiler/run_profiles.sh (same cooldown limits, timeout and outputs).
#
# SETTINGS (.env, written by setup-config.ps1): ANDROID_SDK_ROOT
#
# USAGE (from anywhere):
#   .\profiler\run_profiles.ps1 [-HtpProfile 0|1|2] [-ResultsDir DIR] <folder>=<config> ...
#
# -HtpProfile (default 0) is passed to run-profile-test.ps1; 0 is clean timing.
# Example: .\run_profiles.ps1 GPU-Qwen2VL-Clean=config_gpu_qwen2vl.json CPU-Qwen2VL-Clean=config_cpu_qwen2vl.json
#          .\run_profiles.ps1 -HtpProfile 1 HTP-1Core-ViT-Qwen2VL-Prof=config_htp_1core_vit_qwen2vl.json
# =============================================================================

[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$HtpProfile = $(if ($env:HTP_PROFILE) { [int]$env:HTP_PROFILE } else { 0 }),
    [string]$ResultsDir = $(if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { "Results" }),  # relative to profiler\
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Pairs
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# .env stores Unix-style paths (/c/path) for bash; convert them for PowerShell
function ConvertTo-WindowsPath {
    param([string]$Path)
    if ($Path -match '^/([a-zA-Z])/(.*)$') { return "$($matches[1].ToUpper()):\" + ($matches[2] -replace '/', '\') }
    return $Path
}

# Settings from .env (written by setup-config.ps1 / setup-config.sh); a value the caller already set wins
$EnvFile = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.TrimEnd("`r")
        if ($line -match '^([A-Z_][A-Z0-9_]*)=(.*)$' -and -not [Environment]::GetEnvironmentVariable($matches[1])) {
            Set-Item -Path "Env:\$($matches[1])" -Value $matches[2]
        }
    }
}
if (-not $env:ANDROID_SDK_ROOT) {
    Write-Host "ANDROID_SDK_ROOT is not set: run setup-config.ps1 (it writes .env)" -ForegroundColor Red
    exit 1
}
$Adb = ConvertTo-WindowsPath "$env:ANDROID_SDK_ROOT/platform-tools/adb.exe"
$DeviceDir = "/data/local/tmp/mtmd-profile-test"
$Pwsh = (Get-Process -Id $PID).Path
$TimeoutSec = 3600
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

# Files run-profile-test.ps1 leaves in profiler\; all of them move into the result folder
$Outputs = @("profile_results*.csv", "profile_log.txt", "profile_log_utf8.txt", "thermal_report.txt",
    "cl_profiling.csv", "cl_profile_report.txt", "cl_trace.json",
    "htp_official_profile_report.txt", "htp_block_profile_report.txt")

# Temperature (millidegrees) of the first thermal zone of this type, or $null
function Get-ZoneTemp {
    param([string]$Type)
    # Single-quoted so $z and $(...) reach the phone shell unexpanded
    $cmd = 'for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = {0} ] && cat $z/temp && break; done' -f $Type
    $t = & $Adb shell $cmd 2>$null
    $t = "$t".Trim()
    if ($t -match '^\d+$') { return [int]$t } else { return $null }
}

function Wait-Cooldown {
    for ($i = 0; $i -lt 30; $i++) {
        $gpu = Get-ZoneTemp "gpuss-0"; $npu = Get-ZoneTemp "nsphvx-0"; $cpu = Get-ZoneTemp "cpu-0-0-0"
        Write-Host "$(Get-Date -Format HH:mm:ss) cooldown gpu=$gpu npu=$npu cpu=$cpu"
        $g = if ($null -ne $gpu) { $gpu } else { 99000 }
        $n = if ($null -ne $npu) { $npu } else { 99000 }
        $c = if ($null -ne $cpu) { $cpu } else { 99000 }
        if ($g -lt 44000 -and $n -lt 44000 -and $c -lt 47000) { return }
        Start-Sleep -Seconds 30
    }
    Write-Host "WARNING: cooldown limit reached, phone still warm"
}

foreach ($pair in $Pairs) {
    $folder, $config = $pair -split '=', 2
    $name = Join-Path $ResultsDir $folder
    if (Test-Path $name) { Write-Host "SKIP ${name}: folder exists"; continue }
    Wait-Cooldown
    New-Item -ItemType Directory -Path $name | Out-Null
    Copy-Item "configs\$config" $name
    "htp_profile=$HtpProfile" | Set-Content "$name\run_settings.txt"
    # Stale outputs: local ones from an earlier run, and the OpenCL files the backend writes
    # in its working directory (run-profile-test.ps1 also clears them)
    Remove-Item $Outputs -ErrorAction SilentlyContinue
    & $Adb shell "rm -f $DeviceDir/cl_profiling.csv $DeviceDir/cl_trace.json" 2>$null | Out-Null
    Write-Host "=== $(Get-Date -Format HH:mm:ss) START $name ($config, htp_profile=$HtpProfile)"

    # Separate pwsh with all output streams in console.txt; stopped after $TimeoutSec (exit 124, like timeout)
    $console = (Join-Path (Resolve-Path $name).Path "console.txt")
    $cmd = "& '.\run-profile-test.ps1' -Config 'configs/$config' -HtpProfile $HtpProfile *> '$console'; exit `$LASTEXITCODE"
    $proc = Start-Process -FilePath $Pwsh -ArgumentList @("-NoProfile", "-Command", $cmd) -NoNewWindow -PassThru
    if ($proc.WaitForExit($TimeoutSec * 1000)) {
        $code = $proc.ExitCode
    } else {
        & taskkill /T /F /PID $proc.Id 2>$null | Out-Null
        $code = 124
    }

    Get-ChildItem $Outputs -ErrorAction SilentlyContinue | Move-Item -Destination $name -Force
    $log = (Get-Content "$name\$config" -Raw | ConvertFrom-Json).output_log
    & $Adb pull $log "$name\" 2>$null | Out-Null
    $rows = (Get-ChildItem "$name\profile_results*.csv" -ErrorAction SilentlyContinue | Get-Content | Measure-Object -Line).Lines
    Write-Host "=== $(Get-Date -Format HH:mm:ss) END $name exit=$code csv_lines=$rows files: $((Get-ChildItem $name).Name -join ' ')"
}
Write-Host "ALL DONE"
