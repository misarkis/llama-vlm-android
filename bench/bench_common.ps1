# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

# =============================================================================
# bench_common.ps1
# =============================================================================
# Shared by bench\run_public.ps1, bench\run_screenqa_short.ps1, bench\run_mmbench_gui.ps1
# and bench\setup_eval.ps1 (dot-sourced). The .sh versions carry the same logic inline.
#
#   - loads .env (written by setup-config.ps1 / setup-config.sh); a value already set wins
#   - Assert-Settings: stops with a clear message when a required .env key is missing
#   - $BenchModels: the model table (same files as the .sh scripts)
#   - Start-BenchServer / Stop-BenchServer: one llama-server per model, thinking off
# =============================================================================

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# .env stores Unix-style paths (/c/path) for bash; convert them for PowerShell
function ConvertTo-WindowsPath {
    param([string]$Path)
    if ($Path -match '^/([a-zA-Z])/(.*)$') {
        return "$($matches[1].ToUpper()):\" + ($matches[2] -replace '/', '\')
    }
    return $Path
}

$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.TrimEnd("`r")
        if ($line -match '^([A-Z_][A-Z0-9_]*)=(.*)$') {
            $key = $matches[1]; $value = $matches[2]
            if (-not [Environment]::GetEnvironmentVariable($key)) {
                Set-Item -Path "Env:\$key" -Value (ConvertTo-WindowsPath $value)
            }
        }
    }
}

function Assert-Settings {
    param([string[]]$Keys)
    foreach ($k in $Keys) {
        if (-not [Environment]::GetEnvironmentVariable($k)) {
            Write-Host "$k is not set: run setup-config.ps1 (it writes .env)" -ForegroundColor Red
            exit 1
        }
    }
}

# name -> llama-server model args (paths relative to MODEL_DIR, or -hf repo:quant)
function Get-BenchModels {
    $d = $env:MODEL_DIR
    return [ordered]@{
        "qwen35_0.8b"    = @("-m", "$d/Qwen3.5-VL-0.8B-Unsloth/Qwen3.5-0.8B-Q4_K_M.gguf", "--mmproj", "$d/Qwen3.5-VL-0.8B-Unsloth/mmproj-F16.gguf")
        "qwen38_2b"      = @("-m", "$d/Qwen3.8-2B/Qwen3.8-2B-Distill-heretic.Q4_K_M.gguf", "--mmproj", "$d/Qwen3.8-2B/Qwen3.8-2B-Distill-heretic.mmproj-f16.gguf")
        "qwen35_2b"      = @("-m", "$d/Qwen3.5-VL-2B-Unsloth/Qwen3.5-2B-Q4_K_M.gguf", "--mmproj", "$d/Qwen3.5-VL-2B-Unsloth/mmproj-F16.gguf")
        "qwen3vl_2b"     = @("-m", "$d/Qwen3-VL-2B-Unsloth/Qwen3-VL-2B-Instruct-Q4_K_M.gguf", "--mmproj", "$d/Qwen3-VL-2B-Unsloth/mmproj-F16.gguf")
        "qwen2vl_2b"     = @("-m", "$d/Qwen2-VL-2B-Bartowski/Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "--mmproj", "$d/Qwen2-VL-2B-Bartowski/mmproj-Qwen2-VL-2B-Instruct-f16.gguf")
        "qwen35_4b"      = @("-m", "$d/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf", "--mmproj", "$d/Qwen3.5-4B/mmproj-f16.gguf")
        "lfm25_1.6b"     = @("-m", "$d/LFM2.5-VL-1.6B/LFM2.5-VL-1.6B-Q4_K_M.gguf", "--mmproj", "$d/LFM2.5-VL-1.6B/mmproj-LFM2.5-VL-1.6B-F16.gguf")
        "smolvlm2_2.2b"  = @("-m", "$d/SmolVLM2-2.2B/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf", "--mmproj", "$d/SmolVLM2-2.2B/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf")
        "granite32_2b"   = @("-m", "$d/GraniteVision3.2-2B/granite-vision-3.2-2b-Q4_K_M.gguf", "--mmproj", "$d/GraniteVision3.2-2B/mmproj-model-f16.gguf")
        # Reference row (plan: official Qwen3.5-2B at BF16); downloaded on first use
        "qwen35_2b_bf16" = @("-hf", "unsloth/Qwen3.5-2B-GGUF:BF16")
    }
}
$DefaultOrder = "qwen35_0.8b qwen38_2b qwen35_2b qwen3vl_2b qwen2vl_2b qwen35_4b lfm25_1.6b smolvlm2_2.2b granite32_2b"

# Quote each argument for Start-Process (it takes one command-line string)
function Join-ProcessArgs {
    param([string[]]$ArgList)
    ($ArgList | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
}

function Test-PortInUse {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $client.Dispose() }
}

function Test-ServerReady {
    param([int]$Port)
    try { return ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq "ok") } catch { return $false }
}

# Starts llama-server for one model and waits (max 10 min) for /health = ok.
# Returns the process, or $null if it did not come up (log in $Log).
function Start-BenchServer {
    param([string[]]$ModelArgs, [int]$Port, [int]$Ctx, [int]$Slots, [string]$Log)
    # Thinking off (Qwen3.5/3.8 templates read enable_thinking; other templates ignore it). Greedy comes from the client.
    $serverArgs = $ModelArgs + @("-c", "$Ctx", "-np", "$Slots", "-ngl", "99", "--jinja",
        "--reasoning", "off", "--host", "127.0.0.1", "--port", "$Port")
    $proc = Start-Process -FilePath $env:LLAMA_SERVER -ArgumentList (Join-ProcessArgs $serverArgs) `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError $Log -NoNewWindow -PassThru
    for ($i = 0; $i -lt 300; $i++) {
        if (Test-ServerReady $Port) { return $proc }
        if ($proc.HasExited) { break }
        Start-Sleep -Seconds 2
    }
    Stop-BenchServer $proc
    return $null
}

function Stop-BenchServer {
    param($Proc)
    if ($Proc -and -not $Proc.HasExited) {
        Stop-Process -Id $Proc.Id -ErrorAction SilentlyContinue
        $Proc.WaitForExit(30000) | Out-Null
    }
}

# Python inside a venv created by setup_eval (Windows or Unix layout)
function Get-VenvPython {
    param([string]$Venv)
    foreach ($p in @("$Venv\Scripts\python.exe", "$Venv/bin/python")) { if (Test-Path $p) { return (Resolve-Path $p).Path } }
    Write-Host "No Python venv at $Venv`: run bench\setup_eval.ps1 first" -ForegroundColor Red
    exit 1
}
