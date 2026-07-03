# run-profile-test.ps1
# Deploys and runs mtmd-profile-test on Android device with HTP profiling enabled
#
# Usage: .\run-profile-test.ps1 [-Config <path>] [-HtpProfile <0|1|2>] [-Help]
#
# Environment variables set BEFORE running:
#   GGML_HEXAGON_PROFILE=1  - Enable per-operation HTP profiling (REQUIRED for profiling output)
#   GGML_HEXAGON_NDEV=1     - Number of HTP devices/cores
#   LD_LIBRARY_PATH=.       - Find shared libraries in current directory
#
# Environment variables from .env (auto-loaded):
#   ANDROID_SDK_ROOT    - Path to Android SDK (for ADB)
#   ANDROID_NDK_HOME    - Path to Android NDK (for libomp.so)
param(
    [string]$Config = "",
    [int]$HtpProfile = 1,  # 0=off, 1=basic, 2=extended (must be set via env var before exec)
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\run-profile-test.ps1 [-Config <path>] [-HtpProfile <0|1|2>]"
    Write-Host ""
    Write-Host "IMPORTANT: GGML_HEXAGON_PROFILE must be set BEFORE the process starts"
    Write-Host "           This script sets it automatically."
    Write-Host ""
    Write-Host "GPU Profiling:"
    Write-Host "  To enable OpenCL GPU profiling, rebuild with:"
    Write-Host "    ./build-llamacpp-android.sh -Profile"
    Write-Host "  (Default build has no GPU profiling overhead)"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Config       Path to JSON config file (default: configs/config_htp_1core.json)"
    Write-Host "  -HtpProfile   HTP profiling level: 0=off, 1=basic, 2=extended (default: 1)"
    Write-Host "  -Help         Show this help"
    Write-Host ""
    Write-Host "Example:"
    Write-Host "  .\run-profile-test.ps1 -Config configs/config_htp_1core.json -HtpProfile 1"
    exit 0
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProfilerDir = $ScriptDir
$LlamaCppDir = Join-Path $ScriptDir "..\llama.cpp"
$BuildDirPath = Join-Path $LlamaCppDir "build-android"

# Load environment configuration from .env
$EnvFile = Join-Path $ScriptDir "..\.env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            Set-Item -Force -Path "ENV:\$($matches[1])" -Value $matches[2]
        }
    }
}

# Helper function to convert Unix path to Windows path
function ConvertTo-WindowsPath {
    param([string]$Path)
    if ($Path -match '^/([a-z])/') {
        $drive = $matches[1].ToUpper()
        $rest = $Path.Substring(3) -replace '/', '\'
        return "${drive}:\" + $rest
    }
    return $Path -replace '/', '\'
}

# ADB path from ANDROID_SDK_ROOT
$AdbPath = if ($env:ANDROID_SDK_ROOT) {
    ConvertTo-WindowsPath -Path "$env:ANDROID_SDK_ROOT\platform-tools\adb.exe"
} else {
    "C:\Users\misar\AppData\Local\Android\Sdk\platform-tools\adb.exe"
}

# OMP LIB path from ANDROID_NDK_HOME
$OMP_LIB = if ($env:ANDROID_NDK_HOME) {
    ConvertTo-WindowsPath -Path "$env:ANDROID_NDK_HOME\toolchains\llvm\prebuilt\windows-x86_64\lib64\clang\14.0.6\lib\linux\aarch64\libomp.so"
} else {
    "c:\Users\misar\AppData\Local\Android\Sdk\ndk\25.1.8937393\toolchains\llvm\prebuilt\windows-x86_64\lib64\clang\14.0.6\lib\linux\aarch64\libomp.so"
}

# Default config
if ([string]::IsNullOrEmpty($Config)) {
    $Config = Join-Path $ProfilerDir "configs\config_htp_1core.json"
}

# Device paths
$DeviceDir = "/data/local/tmp/mtmd-profile-test"
$LibDir = "$DeviceDir"  # All libs in same dir as executable

# Colors
$Blue = "`e[34m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$NC = "`e[0m"

function Write-Info { Write-Host "${Blue}[INFO]${NC} $args" }
function Write-Success { Write-Host "${Green}[SUCCESS]${NC} $args" }
function Write-Warn { Write-Host "${Yellow}[WARN]${NC} $args" }
function Write-Error { Write-Host "${Red}[ERROR]${NC} $args" }

# Step 0: Clean existing profile data from device
Write-Info "Cleaning existing profile data from device..."
& $AdbPath shell "rm -f /sdcard/profile_results.csv" 2>&1 | Out-Null
& $AdbPath shell "rm -f /sdcard/profile_results_*.csv" 2>&1 | Out-Null
& $AdbPath shell "rm -f /sdcard/profile_log.txt" 2>&1 | Out-Null
& $AdbPath shell "rm -f /sdcard/profile_*_log.txt" 2>&1 | Out-Null
& $AdbPath shell "rm -f /data/local/tmp/cl_profiling.csv" 2>&1 | Out-Null
& $AdbPath shell "rm -f /data/local/tmp/profile_results.csv" 2>&1 | Out-Null
Write-Success "Cleaned existing profile data"

# Step 1: Check if build outputs exist
Write-Info "Checking build outputs..."

$TestExe = Join-Path $BuildDirPath "bin\mtmd-profile-test"

if (-not (Test-Path $TestExe)) {
    Write-Error "mtmd-profile-test executable not found at $TestExe"
    Write-Info "Run .\build-profile-test.ps1 first"
    exit 1
}

Write-Success "Found mtmd-profile-test executable"

# Step 2: Create device directory
Write-Info "Creating device directory $DeviceDir..."
& $AdbPath shell "mkdir -p $DeviceDir" 2>&1 | Out-Null
& $AdbPath shell "mkdir -p $DeviceDir/test_images" 2>&1 | Out-Null
Write-Success "Created directories"

# Step 3: Push test executable
Write-Info "Pushing mtmd-profile-test to device..."
& $AdbPath push $TestExe $DeviceDir/ 2>&1 | Out-Null
& $AdbPath shell "chmod +x $DeviceDir/mtmd-profile-test" 2>&1 | Out-Null
Write-Success "Pushed executable"

# Step 4: Push config file
Write-Info "Pushing config file..."
& $AdbPath push $Config $DeviceDir/config.json 2>&1 | Out-Null
Write-Success "Pushed config"

# Step 5: Push test images
Write-Info "Pushing test images..."
$TestImagesDir = Join-Path $ProfilerDir "test_images"
if (Test-Path $TestImagesDir) {
    # Get list of images
    $Images = Get-ChildItem $TestImagesDir "*.png" -ErrorAction SilentlyContinue
    foreach ($Img in $Images) {
        & $AdbPath push $Img.FullName "$DeviceDir/test_images/" 2>&1 | Out-Null
    }
    Write-Success "Pushed $($Images.Count) image(s)"
} else {
    Write-Warn "Test images directory not found: $TestImagesDir"
}

# Step 6: Push required shared libraries
Write-Info "Pushing shared libraries..."

$Libs = @(
    "bin/libllama.so",
    "bin/libmtmd.so",
    "bin/libggml.so",
    "bin/libggml-base.so",
    "bin/libggml-cpu.so",
    "bin/libggml-opencl.so",
    "bin/libggml-hexagon.so",
    "bin/libllama-common.so"
)

foreach ($Lib in $Libs) {
    $LibPath = Join-Path $BuildDirPath $Lib
    if (Test-Path $LibPath) {
        & $AdbPath push $LibPath $DeviceDir/ 2>&1 | Out-Null
        Write-Info "  Pushed $Lib"
    } else {
        Write-Warn "  Library not found: $Lib"
    }
}


$HTPLibs = @(
    "../lib/libggml-htp-v73.so",
    "../lib/libggml-htp-v75.so",
    "../lib/libggml-htp-v79.so",
    "../lib/libggml-htp-v81.so"
)

foreach ($Lib in $HTPLibs) {
    $LibPath = Join-Path $BuildDirPath $Lib
    if (Test-Path $LibPath) {
        & $AdbPath push $LibPath $DeviceDir/ 2>&1 | Out-Null
        Write-Info "  Pushed $Lib"
    } else {
        Write-Warn "  Library not found: $Lib"
    }
}

# Also push libOMP if it exists
if (Test-Path $OMP_LIB) {
        & $AdbPath push $OMP_LIB $DeviceDir/ 2>&1 | Out-Null
        Write-Info "  Pushed $OMP_LIB"
    } else {
        Write-Warn "  Library not found: $OMP_LIB"
    }

Write-Success "Pushed libraries"

Write-Info "Clearing Old LOGCAT on device..."

# Step 7: Run with HTP profiling ENABLED
# CRITICAL: GGML_HEXAGON_PROFILE must be set BEFORE the process starts
# This is done by exporting it in the shell before running the executable

Write-Info ""
Write-Info "Running mtmd-profile-test with HTP profiling..."
Write-Info "  GGML_HEXAGON_PROFILE=$HtpProfile"
Write-Info "  Config: config.json"
Write-Info ""
Write-Info "========================================"

& $AdbPath shell "logcat -c" 2>&1 | Out-Null

# Export environment variables and run the test
# Note: GGML_HEXAGON_PROFILE must be set BEFORE the process starts for HTP profiling
# GGML_HEXAGON_VERBOSE=1 shows ops from all backends (CPU, GPU, HTP)
$Cmd = "cd $DeviceDir; ./mtmd-profile-test --config config.json"

Write-Info "Command: $Cmd"
Write-Info ""

& $AdbPath shell "$Cmd" 2>&1

Write-Info ""
Write-Info "========================================"
Write-Info ""

# Step 8: Pull results
$ResultsFile = Join-Path $ProfilerDir "profile_results.csv"
$LogFile = Join-Path $ProfilerDir "profile_log.txt"
$ClProfileFile = Join-Path $ProfilerDir "cl_profiling.csv"

Write-Info "Pulling profile_results.csv from device..."
& $AdbPath pull "/sdcard/profile_results.csv" $ResultsFile 2>&1 | Out-Null
if (Test-Path $ResultsFile) {
    Write-Success "Results pulled to: $ResultsFile"
}

Write-Info "Running LOGCAT on device..."
& $AdbPath shell "logcat -s -d mtmd-profile-test:V mtmd-inference-api:V *:F> /sdcard/profile_log.txt" 2>&1 | Out-Null
& $AdbPath pull "/sdcard/profile_log.txt" $LogFile 2>&1 | Out-Null
if (Test-Path $LogFile) {
    Write-Success "Log pulled to: $LogFile"
}

# Pull OpenCL profiling data (if built with GGML_OPENCL_PROFILING=ON)
Write-Info "Pulling cl_profiling.csv from device..."
& $AdbPath pull "$DeviceDir/cl_profiling.csv" $ClProfileFile 2>&1 | Out-Null
if (Test-Path $ClProfileFile) {
    $Size = (Get-Item $ClProfileFile).Length
    Write-Success "OpenCL profiling data pulled to: $ClProfileFile ($([math]::Round($Size/1KB, 2)) KB)"
} else {
    Write-Warn "No OpenCL profiling data found (rebuild with GGML_OPENCL_PROFILING=ON to enable)"
}

Write-Success "Profile test completed"
Write-Info ""

# Step 9: Generate profile reports
# Generate HTP reports if log file exists
if (Test-Path $LogFile) {
    Write-Info "Generating HTP profile reports..."

    # Convert log to UTF-8 for Python scripts (Android logcat outputs UTF-8 on most devices)
    $LogFileUtf8 = Join-Path $ProfilerDir "profile_log_utf8.txt"
    Write-Info "  Converting log to UTF-8..."
    # Try UTF-8 first (most common), fall back to UTF-16
    try {
        Get-Content $LogFile -Encoding UTF8 -ErrorAction Stop | Set-Content $LogFileUtf8 -Encoding UTF8
    } catch {
        Get-Content $LogFile -Encoding Unicode | Set-Content $LogFileUtf8 -Encoding UTF8
    }

    # Generate official ggml-hexagon-profile.py report
    $OfficialReport = Join-Path $ProfilerDir "htp_official_profile_report.txt"
    Write-Info "  Generating official HTP profile report (ggml-hexagon-profile.py)..."
    $LlamaCppScriptsDir = Join-Path $ScriptDir "..\llama.cpp\scripts\snapdragon"
    python "$LlamaCppScriptsDir\ggml-hexagon-profile.py" $LogFileUtf8 -n 100 --sort max-usec > $OfficialReport 2>&1
    if (Test-Path $OfficialReport) {
        $Size = (Get-Item $OfficialReport).Length
        if ($Size -gt 0) {
            Write-Success "Official report generated: $OfficialReport ($([math]::Round($Size/1KB, 2)) KB)"
        } else {
            Write-Warn "Official report is empty - check if log contains HTP profiling data"
        }
    } else {
        Write-Warn "Failed to generate official report"
    }

    # Generate block-grouped report (parse_profile.py)
    $BlockReport = Join-Path $ProfilerDir "htp_block_profile_report.txt"
    Write-Info "  Generating block-grouped profile report (parse_profile.py)..."
    & python "$ScriptDir\parse_profile.py" $LogFileUtf8 $BlockReport
    if (Test-Path $BlockReport) {
        Write-Success "Block report generated: $BlockReport"
    } else {
        Write-Warn "Failed to generate block report"
    }

    Write-Info ""
    Write-Info "HTP Profile reports generated:"
    Write-Info "  - Official (op-level): $OfficialReport"
    Write-Info "  - Block-grouped:       $BlockReport"
} else {
    Write-Warn "Log file not found, skipping HTP report generation"
}

# Generate OpenCL report if cl_profiling.csv exists
if (Test-Path $ClProfileFile) {
    Write-Info ""
    Write-Info "Generating OpenCL profile report..."

    $ClReport = Join-Path $ProfilerDir "cl_profile_report.txt"
    Write-Info "  Generating OpenCL GPU profile report (parse_profile.py)..."
    & python "$ScriptDir\parse_profile.py" $ClProfileFile $ClReport --mode opencl
    if (Test-Path $ClReport) {
        Write-Success "OpenCL report generated: $ClReport"
    } else {
        Write-Warn "Failed to generate OpenCL report"
    }
} else {
    Write-Warn "No OpenCL profiling data found, skipping OpenCL report"
}

Write-Info ""
Write-Info "To analyze profiling output manually:"
Write-Info "  HTP Official:      python ../llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py <log_file>"
Write-Info "  HTP Block Report:  python parse_profile.py <log_file> [output_file]"
Write-Info "  OpenCL Report:     python parse_profile.py <cl_profiling.csv> [output_file] --mode opencl"
Write-Info ""
