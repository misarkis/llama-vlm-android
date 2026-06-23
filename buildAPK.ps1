# =============================================================================
# buildAPK.ps1
# =============================================================================
# Builds the VLM-Analyze Android APK with all dependencies.
#
# BUILD FLOW:
#   -All:      Build llama.cpp -> mtmd-inference-api -> sherpa-onnx -> APK
#   Default:   Build APK only (uses existing libraries if present)
#
# USAGE:
#   .\buildAPK.ps1                     # Build APK only (debug, auto-detect deps)
#   .\buildAPK.ps1 -All                # Full rebuild: all deps + APK
#   .\buildAPK.ps1 -All -install       # Full rebuild + install on device
#   .\buildAPK.ps1 -install            # Build APK + install on device
#   .\buildAPK.ps1 -release            # Build release APK
#   .\buildAPK.ps1 -clean              # Clean Gradle build artifacts
#   .\buildAPK.ps1 -Help               # Show this help
#
# OUTPUT:
#   Debug APK:  app\build\outputs\apk\debug\app-debug.apk
#   Release:    app\build\outputs\apk\release\app-release.apk
# =============================================================================

# Load environment configuration if exists
$EnvFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            Set-Item -Force -Path "ENV:\$($matches[1])" -Value $matches[2]
        }
    }
}

# Parse arguments - use $args only (works with both direct invocation and -File)
$All = $false
$Install = $false
$Release = $false
$Clean = $false
$Help = $false

# Check $args for flags
foreach ($arg in $args) {
    switch ($arg) {
        '-All' { $All = $true }
        '-install' { $Install = $true }
        '-release' { $Release = $true }
        '-clean' { $Clean = $true }
        '-Help' { $Help = $true }
    }
}

if ($Help) {
    Write-Host "Usage: .\buildAPK.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -All        Build llama.cpp + mtmd-inference-api + sherpa + APK"
    Write-Host "  -install    Install APK on connected device after build"
    Write-Host "  -release    Build release APK (default: debug)"
    Write-Host "  -clean      Clean Gradle build artifacts"
    Write-Host "  -Help       Show this help"
    exit 0
}

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Helper function to convert Unix path to Windows path (for .env format: /c/msys64 -> C:\msys64)
function ConvertTo-WindowsPath {
    param([string]$Path)
    # Handle Unix format: /c/path/to/something
    if ($Path -match '^/([a-z])/') {
        $drive = $matches[1].ToUpper()
        $rest = $Path.Substring(3) -replace '/', '\'
        return "${drive}:\" + $rest
    }
    # Already Windows format or other format - normalize slashes
    return $Path -replace '/', '\'
}

# Helper function to get MSYS2 bash path
function Get-MSYS2Bash {
    if ($env:MSYS2_PATH) {
        $Msys2Path = ConvertTo-WindowsPath -Path $env:MSYS2_PATH
        return "$Msys2Path\usr\bin\bash.exe"
    } else {
        Write-Host "MSYS2_PATH not set in .env. Please run setup-config.ps1 first."
        return $null
    }
}

# Helper function to set MSYS2 PATH
function Set-MSYS2Path {
    if ($env:MSYS2_PATH) {
        $Msys2Path = ConvertTo-WindowsPath -Path $env:MSYS2_PATH
        $env:PATH = "$Msys2Path\usr\bin;$Msys2Path\ucrt64\bin;$($env:PATH)"
    }
}

# Gradle wrapper path
$GradleWrapper = Join-Path $ScriptDir "gradlew.bat"

if (-not (Test-Path $GradleWrapper)) {
    Write-Host "Gradle wrapper not found. Using system Gradle..."
    $Gradle = "gradle"
} else {
    $Gradle = $GradleWrapper
}

# Helper functions
function Test-LlamaCppLibs {
    $JniLibsDir = Join-Path $ScriptDir "app\src\main\jniLibs\arm64-v8a"
    return Test-Path "$JniLibsDir\libllama.so"
}

function Test-MtmdInferenceApi {
    $MtmdLib = Join-Path $ScriptDir "app\src\main\cpp\libs\arm64-v8a\libmtmd-inference-api.a"
    return Test-Path $MtmdLib
}

# Clean action
if ($Clean) {
    Write-Host "Cleaning VLM-Analyze Android..."
    & $Gradle clean
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Gradle clean failed (files may be locked). Attempting manual cleanup..."
        Remove-Item -Recurse -Force "app\build" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force ".gradle" -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force "app\.cxx" -ErrorAction SilentlyContinue
        Write-Host "Manual cleanup complete."
    } else {
        Write-Host "Clean successful!"
    }
    exit 0
}

# Determine build configuration
$Config = if ($Release) { "Release" } else { "Debug" }
$BuildTask = "assemble$Config"

# Step 1: Build llama.cpp if -All flag is set
if ($All) {
    Write-Host "=============================================="
    Write-Host "Building llama.cpp (CPU + OpenCL + Hexagon)..."
    Write-Host "=============================================="

    $MSYS2Bash = Get-MSYS2Bash
    if (-not $MSYS2Bash) { exit 1 }
    Set-MSYS2Path
    & $MSYS2Bash -c "cd '$ScriptDir' && ./build-llamacpp-android.sh"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: llama.cpp build failed"
        exit 1
    }
    Write-Host "llama.cpp build complete!"
} else {
    # Auto-detect: build llama.cpp if libraries missing
    if (-not (Test-LlamaCppLibs)) {
        Write-Host "llama.cpp libraries not found in jniLibs..."
        Write-Host "Running build-llamacpp-android.sh..."

        $MSYS2Bash = Get-MSYS2Bash
        if (-not $MSYS2Bash) { exit 1 }
        Set-MSYS2Path
        & $MSYS2Bash -c "cd '$ScriptDir' && ./build-llamacpp-android.sh"

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: llama.cpp build failed"
            exit 1
        }
        Write-Host "llama.cpp build complete!"
    }
}

# Step 2: Build mtmd-inference-api if -All flag is set
if ($All) {
    Write-Host "=============================================="
    Write-Host "Building mtmd-inference-api..."
    Write-Host "=============================================="

    $MtmdScript = Join-Path $ScriptDir "build-mtmd-inference-api.ps1"
    if (Test-Path $MtmdScript) {
        & $MtmdScript -All
    } else {
        Write-Host "ERROR: build-mtmd-inference-api.ps1 not found"
        exit 1
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: mtmd-inference-api build failed"
        exit 1
    }
    Write-Host "mtmd-inference-api build complete!"
} else {
    # Auto-detect: build mtmd-inference-api if static lib missing
    if (-not (Test-MtmdInferenceApi)) {
        Write-Host "mtmd-inference-api not found..."
        Write-Host "Running build-mtmd-inference-api.ps1..."

        $MtmdScript = Join-Path $ScriptDir "build-mtmd-inference-api.ps1"
        if (Test-Path $MtmdScript) {
            & $MtmdScript
        } else {
            Write-Host "ERROR: build-mtmd-inference-api.ps1 not found"
            Write-Host "Please run build-mtmd-inference-api.ps1 manually first."
            exit 1
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: mtmd-inference-api build failed"
            exit 1
        }
        Write-Host "mtmd-inference-api build complete!"
    }
}

# Step 3: Build sherpa-onnx if -All flag is set
if ($All) {
    Write-Host "=============================================="
    Write-Host "Building sherpa-onnx..."
    Write-Host "=============================================="

    $SherpaScript = Join-Path $ScriptDir "build-sherpa-from-source.ps1"
    if (Test-Path $SherpaScript) {
        & $SherpaScript
    } else {
        Write-Host "Info: build-sherpa-from-source.ps1 not found - skipping sherpa build"
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: sherpa-onnx build failed (may already be built)"
    } else {
        Write-Host "sherpa-onnx build complete!"
    }

    # Ensure we're back in the script directory (sherpa build may have changed it)
    Set-Location $ScriptDir
}

# Step 4: Build the APK
Write-Host "=============================================="
Write-Host "Building VLM-Analyze Android ($Config)..."
Write-Host "=============================================="

& $Gradle $BuildTask

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "Build successful!"
    Write-Host "=============================================="

    if ($Config -eq "Debug") {
        $ApkPath = "app\build\outputs\apk\debug\app-debug.apk"
        Write-Host "Debug APK: $ApkPath"

        if ($Install) {
            Write-Host "Installing on connected device..."
            & $Gradle installDebug
        }
    } else {
        $ApkPath = "app\build\outputs\apk\release\app-release.apk"
        Write-Host "Release APK: $ApkPath"
        Write-Host ""
        Write-Host "To install manually: adb install $ApkPath"
    }
    Write-Host ""
} else {
    Write-Host "Build failed (files may be locked by Gradle/Android Studio)."
    Write-Host "Try: Get-Process java -ErrorAction SilentlyContinue | Stop-Process -Force"
    exit 1
}
