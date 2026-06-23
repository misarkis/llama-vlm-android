# =============================================================================
# build-mtmd-inference-api.ps1
# =============================================================================
# Builds the mtmd-inference-api static library for Android and prepares
# all integration files needed for the VLM-Analyze Android app.
#
# PREREQUISITES:
#   - MSYS2 UCRT64 environment with bash
#   - Android NDK 25.1.8937393 (set via ANDROID_NDK_HOME env var)
#   - CMake 3.21+
#
# WHAT THIS SCRIPT DOES (default, no flags):
#   1. Copies mtmd-inference-api source to llama.cpp/tools/
#   2. Updates llama.cpp/tools/CMakeLists.txt to include the new subdirectory
#   3. Builds libmtmd-inference-api.a (static library for Android)
#   4. Builds mtmd-inference-test (test executable)
#   5. Copies integration files to app/src/main/cpp/:
#      - mtmd-inference-api.h (public API header)
#      - libmtmd-inference-api.a (static library)
#      - llama.h, ggml*.h (required headers)
#
# WITH -All:
#   Also rebuilds llama.cpp from scratch (CPU + OpenCL + Hexagon backends)
#   before building mtmd-inference-api.
#
# OUTPUT FILES:
#   Shared libs (from build-llamacpp-android.sh):
#     app/src/main/jniLibs/arm64-v8a/
#       - libllama.so, libmtmd.so, libggml*.so, libggml-htp-*.so
#
#   This script:
#     app/src/main/cpp/mtmd-api/
#       - mtmd-inference-api.h (public API)
#     app/src/main/cpp/mtmd-api/include/
#       - llama.h, ggml*.h (required headers)

# Load environment configuration if exists
$EnvFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            Set-Item -Force -Path "ENV:\$($matches[1])" -Value $matches[2]
        }
    }
}
#     app/src/main/cpp/libs/arm64-v8a/
#       - libmtmd-inference-api.a (static library for linking)
#
# USAGE:
#   .\build-mtmd-inference-api.ps1     # Build mtmd-inference-api only
#   .\build-mtmd-inference-api.ps1 -All # Build llama.cpp + mtmd-inference-api
#   .\build-mtmd-inference-api.ps1 -Help
#
# NEXT STEP:
#   After this script completes, build the APK:
#     ./gradlew assembleDebug
#     ./gradlew installDebug
# =============================================================================

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

# Parse arguments - use $args only (works with both direct invocation and -File)
$All = $false
$Help = $false

# CMake path from .env or default to MSYS2 UCRT64 CMake
if ($env:MSYS2_PATH) {
    $Msys2Path = ConvertTo-WindowsPath -Path $env:MSYS2_PATH
    $CmakePath = "$Msys2Path\ucrt64\bin\cmake.exe"
} else {
    $CmakePath = "C:\Program Files\CMake\bin\cmake.exe"
}

# Check $args for flags
$argArray = @($args)
for ($i = 0; $i -lt $argArray.Count; $i++) {
    switch ($argArray[$i]) {
        '-All' { $All = $true }
        '-Help' { $Help = $true }
        '-CmakePath' { if ($i -lt $argArray.Count - 1) { $i++; $CmakePath = $argArray[$i] } }
    }
}

if ($Help) {
    Write-Host "Usage: .\build-mtmd-inference-api.ps1 [-All]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -All  Build llama.cpp + mtmd-inference-api"
    exit 0
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LlamaCppDir = Join-Path $ScriptDir "llama.cpp"
$MtmdSourceDir = Join-Path $ScriptDir "mtmd-inference-api"
$MtmdTargetDir = Join-Path $LlamaCppDir "tools\mtmd-inference-api"
$BuildDirPath = Join-Path $LlamaCppDir "build-android"

# App integration directories
$AppMtmdApiDir = Join-Path $ScriptDir "app\src\main\cpp\mtmd-api"
$AppMtmdApiIncludeDir = Join-Path $AppMtmdApiDir "include"
$AppLibsDir = Join-Path $ScriptDir "app\src\main\cpp\libs\arm64-v8a"

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

# Step 1: Verify source directory exists
Write-Info "Checking mtmd-inference-api source directory..."
if (-not (Test-Path $MtmdSourceDir)) {
    Write-Error "mtmd-inference-api source not found at $MtmdSourceDir"
    exit 1
}
Write-Success "Source directory found"

# Step 2: Copy source to llama.cpp/tools/mtmd-inference-api
Write-Info "Copying mtmd-inference-api to llama.cpp/tools/..."

if (Test-Path $MtmdTargetDir) {
    Write-Info "Removing existing mtmd-inference-api in llama.cpp..."
    Remove-Item $MtmdTargetDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $MtmdTargetDir | Out-Null
Copy-Item $MtmdSourceDir\* -Destination $MtmdTargetDir -Recurse -Force
Write-Success "Copied to $MtmdTargetDir"

# Step 3: Update tools/CMakeLists.txt if needed
Write-Info "Checking llama.cpp integration..."

$ToolsCMakeLists = Join-Path $LlamaCppDir "tools\CMakeLists.txt"
if (Test-Path $ToolsCMakeLists) {
    $Content = Get-Content $ToolsCMakeLists -Raw
    if ($Content -notmatch 'add_subdirectory\(mtmd-inference-api\)') {
        Write-Info "Adding mtmd-inference-api to tools/CMakeLists.txt..."
        # Use string concatenation with actual newline (backtick-n doesn't work in -replace)
        $Match = [regex]::Match($Content, 'add_subdirectory\(mtmd\)')
        if ($Match.Success) {
            $InsertPos = $Match.Index + $Match.Length
            $Content = $Content.Insert($InsertPos, "`n    add_subdirectory(mtmd-inference-api)")
            Set-Content $ToolsCMakeLists $Content -NoNewline
            Write-Success "Updated tools/CMakeLists.txt"
        }
    } else {
        Write-Info "mtmd-inference-api already integrated in CMakeLists.txt"
    }
} else {
    Write-Warn "tools/CMakeLists.txt not found"
}

# Step 4: Clean build if requested
if ($All) {
    Write-Info "Building llama.cpp + mtmd-inference-api..."

    # Rebuild llama.cpp with Android configuration using shell script
    Write-Info "Rebuilding llama.cpp with Android configuration..."

    if ($env:MSYS2_PATH) {
        $Msys2Path = ConvertTo-WindowsPath -Path $env:MSYS2_PATH
        $BashPath = "$Msys2Path\usr\bin\bash.exe"
        if (Test-Path $BashPath) {
            # Set PATH to include MSYS2 tools (git, etc.)
            $env:PATH = "$Msys2Path\usr\bin;$Msys2Path\ucrt64\bin;$($env:PATH)"
            & $BashPath -c "cd '$ScriptDir' && ./build-llamacpp-android.sh"
        } else {
            Write-Error "MSYS2 bash not found at $BashPath"
            exit 1
        }
    } else {
        Write-Error "MSYS2_PATH not set in .env. Please run setup-config.ps1 first."
        exit 1
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "llama.cpp build failed"
        exit 1
    }
    Write-Success "llama.cpp rebuilt successfully"
}

# Step 5: Build mtmd-inference-api
Write-Info "Building mtmd-inference-api library..."

# Note: CMake is pre-configured by build-llamacpp-android.sh
# Do not reconfigure here - it would use wrong parameters for Android
& $CmakePath "--build" $BuildDirPath "--target" "mtmd-inference-api" "mtmd-inference-test" "--" "-j4" 2>&1 | Out-String

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

Write-Success "Build completed!"

# Step 6: Verify build outputs
$StaticLib = Join-Path $BuildDirPath "tools\mtmd-inference-api\libmtmd-inference-api.a"
$TestExe = Join-Path $BuildDirPath "bin\mtmd-inference-test"

Write-Info ""
Write-Info "Build outputs:"
if (Test-Path $StaticLib) {
    $Size = (Get-Item $StaticLib).Length / 1MB
    Write-Info "  Static library: $([math]::Round($Size, 2)) MB"
} else {
    Write-Error "Static library not found at $StaticLib"
    exit 1
}

if (Test-Path $TestExe) {
    $Size = (Get-Item $TestExe).Length / 1MB
    Write-Info "  Test executable: $([math]::Round($Size, 2)) MB"
}

# Step 7: Copy files to app directory for integration
Write-Info ""
Write-Info "Copying files to app directory..."

# Create directories
if (Test-Path $AppMtmdApiDir) {
    Remove-Item $AppMtmdApiDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $AppMtmdApiIncludeDir | Out-Null
New-Item -ItemType Directory -Force -Path $AppLibsDir | Out-Null

# Copy mtmd-inference-api.h
Copy-Item (Join-Path $MtmdSourceDir "mtmd-inference-api.h") $AppMtmdApiDir -Force
Write-Info "  Copied mtmd-inference-api.h"

# Copy static library
Copy-Item $StaticLib $AppLibsDir -Force
Write-Info "  Copied libmtmd-inference-api.a"

# Copy llama.h (required by mtmd-inference-api.h)
$LlamaH = Join-Path $LlamaCppDir "include\llama.h"
if (Test-Path $LlamaH) {
    Copy-Item $LlamaH $AppMtmdApiIncludeDir -Force
    Write-Info "  Copied llama.h"
} else {
    Write-Warn "llama.h not found at $LlamaH"
}

# Copy ggml headers (required by llama.h)
$GgmlIncludeDir = Join-Path $LlamaCppDir "ggml\include"
if (Test-Path $GgmlIncludeDir) {
    Get-ChildItem $GgmlIncludeDir "*.h" | ForEach-Object {
        Copy-Item $_.FullName $AppMtmdApiIncludeDir -Force
        Write-Info "  Copied $($_.Name)"
    }
} else {
    Write-Warn "ggml include directory not found at $GgmlIncludeDir"
}

Write-Success "Files copied to app directory"

# Summary
Write-Info ""
Write-Info "========================================"
Write-Info "  mtmd-inference-api Build Complete"
Write-Info "========================================"
Write-Info ""
Write-Info "App integration files:"
Write-Info "  Headers: $AppMtmdApiIncludeDir"
Write-Info "  Static lib: $AppLibsDir"
Write-Info ""
Write-Info "Shared libraries (already copied by build-llamacpp-android.sh):"
Write-Info "  Location: $ScriptDir\app\src\main\jniLibs\arm64-v8a\"
Write-Info ""
Write-Info "Next: Build the app with ./gradlew assembleDebug"
Write-Info ""

# Ensure we're back in the script directory
Set-Location $ScriptDir
