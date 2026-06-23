# =============================================================================
# VLM-Analyze Android - Setup Configuration (PowerShell)
# =============================================================================
# REQUIRED: MSYS2 UCRT64 is mandatory (llama.cpp builds require it).
# Other dependencies can be configured here, but all builds must run in
# MSYS2 UCRT64 terminal: ./buildAPK.sh -All -install
#
# Run this script ONCE after cloning to create the .env file.
# =============================================================================

Write-Host "=== VLM-Analyze Android Setup ===" -ForegroundColor Cyan
Write-Host "MSYS2 UCRT64 is REQUIRED for building llama.cpp." -ForegroundColor Yellow
Write-Host "All builds must run in MSYS2 UCRT64 terminal." -ForegroundColor Yellow
Write-Host ""

# Clear existing environment variables to force re-prompting
Remove-Item Env:\MSYS2_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\ANDROID_SDK_ROOT -ErrorAction SilentlyContinue
Remove-Item Env:\ANDROID_NDK_HOME -ErrorAction SilentlyContinue
Remove-Item Env:\JAVA_HOME -ErrorAction SilentlyContinue
Remove-Item Env:\HEXAGON_SDK_ROOT -ErrorAction SilentlyContinue

$missing = @()

# Helper function to convert Windows path to Unix format for .env file
function ConvertTo-UnixPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $Path
    }
    # Normalize path: convert forward slashes to backslashes first
    $Path = $Path -replace '/', '\'
    # Remove trailing backslash if present
    $Path = $Path.TrimEnd('\')
    # Handle Windows format: C:\path\to\something
    if ($Path -match '^([A-Za-z]):\\') {
        $drive = $matches[1].ToLower()
        # Substring(3) skips "C:\" to get just the path part
        $rest = $Path.Substring(3) -replace '\\', '/'
        return "/$drive/$rest"
    }
    # Already Unix format - just ensure no double slashes
    return ($Path -replace '/+', '/')
}

# -----------------------------------------------------------------------------
# MSYS2 UCRT64 - REQUIRED (MUST HAVE)
# -----------------------------------------------------------------------------
Write-Host "1. MSYS2 UCRT64 (REQUIRED - Must have for llama.cpp builds):" -ForegroundColor Yellow
if ($env:MSYS2_PATH) {
    Write-Host "   Using existing: $($env:MSYS2_PATH)"
} else {
    $defaultMsys2 = "C:\msys64"
    Write-Host "   Default: $defaultMsys2"
    $input = Read-Host "   Enter MSYS2 path (press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($input)) {
        $env:MSYS2_PATH = $defaultMsys2
    } else {
        $env:MSYS2_PATH = $input
    }
}

if (-not (Test-Path "$env:MSYS2_PATH")) {
    $missing += "MSYS2 not found at: $($env:MSYS2_PATH)"
} elseif (-not (Test-Path "$env:MSYS2_PATH\ucrt64\bin\gcc.exe")) {
    $missing += "MSYS2 UCRT64 toolchain not found. Run: pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake"
} else {
    Write-Host "   [OK] Set: $($env:MSYS2_PATH)" -ForegroundColor Green
}
Write-Host ""

# -----------------------------------------------------------------------------
# ANDROID SDK - REQUIRED
# -----------------------------------------------------------------------------
Write-Host "2. Android SDK (REQUIRED):" -ForegroundColor Yellow
if ($env:ANDROID_SDK_ROOT) {
    Write-Host "   Using existing: $($env:ANDROID_SDK_ROOT)"
} else {
    $defaultSdk = "C:\Users\$env:USERNAME\AppData\Local\Android\Sdk"
    Write-Host "   Default: $defaultSdk"
    $input = Read-Host "   Enter path (press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($input)) {
        $env:ANDROID_SDK_ROOT = $defaultSdk
    } else {
        $env:ANDROID_SDK_ROOT = $input
    }
}
$env:ANDROID_HOME = $env:ANDROID_SDK_ROOT
$env:PATH = "$env:ANDROID_SDK_ROOT\platform-tools;$env:PATH"

if (-not (Test-Path "$env:ANDROID_SDK_ROOT")) {
    $missing += "Android SDK not found at: $($env:ANDROID_SDK_ROOT)"
} else {
    Write-Host "   [OK] Set: $($env:ANDROID_SDK_ROOT)" -ForegroundColor Green
}
Write-Host ""

# -----------------------------------------------------------------------------
# ANDROID NDK - REQUIRED
# -----------------------------------------------------------------------------
Write-Host "3. Android NDK (REQUIRED):" -ForegroundColor Yellow
if ($env:ANDROID_NDK_HOME) {
    Write-Host "   Using existing: $($env:ANDROID_NDK_HOME)"
} else {
    $defaultNdk = "$env:ANDROID_SDK_ROOT\ndk\25.1.8937393"
    Write-Host "   Default: $defaultNdk"
    $input = Read-Host "   Enter path (press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($input)) {
        $env:ANDROID_NDK_HOME = $defaultNdk
    } else {
        $env:ANDROID_NDK_HOME = $input
    }
}

if (-not (Test-Path "$env:ANDROID_NDK_HOME")) {
    $missing += "Android NDK not found at: $($env:ANDROID_NDK_HOME)"
} else {
    Write-Host "   [OK] Set: $($env:ANDROID_NDK_HOME)" -ForegroundColor Green
}
Write-Host ""

# -----------------------------------------------------------------------------
# JAVA JDK - REQUIRED
# -----------------------------------------------------------------------------
Write-Host "4. Java JDK 17+ (REQUIRED):" -ForegroundColor Yellow
if ($env:JAVA_HOME) {
    Write-Host "   Using existing: $($env:JAVA_HOME)"
} else {
    $defaultJdk = "C:\Program Files\Android\Android Studio\jbr"
    Write-Host "   Default (Android Studio JDK): $defaultJdk"
    $input = Read-Host "   Enter path (press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($input)) {
        $env:JAVA_HOME = $defaultJdk
    } else {
        $env:JAVA_HOME = $input
    }
}
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

if (-not (Test-Path "$env:JAVA_HOME")) {
    $missing += "Java JDK not found at: $($env:JAVA_HOME)"
} else {
    Write-Host "   [OK] Set: $($env:JAVA_HOME)" -ForegroundColor Green
}
Write-Host ""

# -----------------------------------------------------------------------------
# HEXAGON SDK - REQUIRED
# -----------------------------------------------------------------------------
Write-Host "5. Hexagon SDK (REQUIRED - for NPU/HTP acceleration):" -ForegroundColor Yellow
if ($env:HEXAGON_SDK_ROOT) {
    Write-Host "   Using existing: $($env:HEXAGON_SDK_ROOT)"
} else {
    $defaultHex = "C:\Hexagon_SDK\6.6.0.0"
    Write-Host "   Default: $defaultHex"
    $input = Read-Host "   Enter path (press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($input)) {
        $env:HEXAGON_SDK_ROOT = $defaultHex
    } else {
        $env:HEXAGON_SDK_ROOT = $input
    }
}

if ([string]::IsNullOrWhiteSpace($env:HEXAGON_SDK_ROOT)) {
    $missing += "Hexagon SDK path is REQUIRED for NPU support"
} elseif (-not (Test-Path "$env:HEXAGON_SDK_ROOT")) {
    $missing += "Hexagon SDK not found at: $($env:HEXAGON_SDK_ROOT)"
} else {
    Write-Host "   [OK] Set: $($env:HEXAGON_SDK_ROOT)" -ForegroundColor Green
}
Write-Host ""

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
Write-Host "=== Validation ===" -ForegroundColor Cyan
if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Missing required dependencies:" -ForegroundColor Red
    foreach ($err in $missing) {
        Write-Host "  [X] $err" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please fix the paths above before continuing." -ForegroundColor Yellow
    Write-Host "Run this script again after correcting the paths." -ForegroundColor Yellow
    exit 1
}

Write-Host "All dependencies found! [OK]" -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
Write-Host "=== Configuration Summary ===" -ForegroundColor Cyan
Write-Host "MSYS2:        $($env:MSYS2_PATH)"
Write-Host "Android SDK:  $($env:ANDROID_SDK_ROOT)"
Write-Host "Android NDK:  $($env:ANDROID_NDK_HOME)"
Write-Host "Java JDK:     $($env:JAVA_HOME)"
Write-Host "Hexagon SDK:  $($env:HEXAGON_SDK_ROOT)"
Write-Host "OpenCL:       [AUTO] Downloaded during build"
Write-Host ""

# Save to .env file (OpenCL is auto-downloaded, no config needed)
# Convert Windows paths to Unix format for bash compatibility
$configContent = @"
MSYS2_PATH=$(ConvertTo-UnixPath -Path $env:MSYS2_PATH)
ANDROID_SDK_ROOT=$(ConvertTo-UnixPath -Path $env:ANDROID_SDK_ROOT)
ANDROID_NDK_HOME=$(ConvertTo-UnixPath -Path $env:ANDROID_NDK_HOME)
JAVA_HOME=$(ConvertTo-UnixPath -Path $env:JAVA_HOME)
HEXAGON_SDK_ROOT=$(ConvertTo-UnixPath -Path $env:HEXAGON_SDK_ROOT)
"@

$configPath = Join-Path $PSScriptRoot ".env"
$configContent | Out-File -FilePath $configPath -Encoding ASCII
Write-Host "Configuration saved to: $configPath" -ForegroundColor Green
Write-Host ""
Write-Host "Next: Open Powershell terminal and run:" -ForegroundColor Cyan
Write-Host "  ./buildAPK.ps1 -All -install" -ForegroundColor White
Write-Host ""
Write-Host "NOTE: The .env file is automatically loaded by all build scripts." -ForegroundColor Gray
Write-Host "You do NOT need to run this script before each build." -ForegroundColor Gray
