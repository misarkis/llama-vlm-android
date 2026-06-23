# =============================================================================
# build-sherpa-from-source.ps1
# =============================================================================
# Builds sherpa-onnx AAR for VLM-Analyze Android integration.
# Uses official pre-built binary tarball from GitHub releases.
#
# PREREQUISITES:
#   - Git for Windows
#   - Android SDK with NDK (r25+ recommended)
#   - Java JDK 17+ (bundled with Android Studio)
#
# WHAT THIS SCRIPT DOES:
#   1. Clones sherpa-onnx repository from GitHub
#   2. Downloads pre-built Android binaries from GitHub releases
#   3. Copies native libraries to Android module jniLibs
#   4. Fixes broken symlinks in Kotlin API
#   5. Downloads Whisper base.en model files
#   6. Builds AAR using Gradle
#   7. Copies AAR to libs/sherpa-onnx.aar
#
# OUTPUT:
#   libs/sherpa-onnx.aar - AAR library for VLM-Analyze Android
#   app/src/main/assets/sherpa-onnx-whisper-base.en/ - Model files
#
# USAGE:
#   .\build-sherpa-from-source.ps1              # Build with default version (v1.13.2)
#   .\build-sherpa-from-source.ps1 v1.13.2      # Build with specific version
#
# NEXT STEP:
#   The AAR is automatically included when building the APK with:
#     .\buildAPK.ps1 -All
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

$ErrorActionPreference = "Stop"

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

# Parse arguments
$VERSION = if ($args[0]) { $args[0] } else { "v1.13.2" }
$SHERPA_DIR = "sherpa-onnx"

# Use ANDROID_SDK_ROOT from .env
if (-not $env:ANDROID_SDK_ROOT) {
    Write-Host "ERROR: ANDROID_SDK_ROOT not set in .env" -ForegroundColor Red
    exit 1
}
$ANDROID_SDK_ROOT = ConvertTo-WindowsPath -Path $env:ANDROID_SDK_ROOT

# Use ANDROID_NDK_HOME from .env
if (-not $env:ANDROID_NDK_HOME) {
    Write-Host "ERROR: ANDROID_NDK_HOME not set in .env" -ForegroundColor Red
    exit 1
}
$env:ANDROID_NDK_ROOT = ConvertTo-WindowsPath -Path $env:ANDROID_NDK_HOME

# Use JAVA_HOME from .env
if (-not $env:JAVA_HOME) {
    Write-Host "ERROR: JAVA_HOME not set in .env" -ForegroundColor Red
    exit 1
}
$env:JAVA_HOME = ConvertTo-WindowsPath -Path $env:JAVA_HOME

Write-Host "Android SDK: $ANDROID_SDK_ROOT" -ForegroundColor Gray
Write-Host "Android NDK: $env:ANDROID_NDK_ROOT" -ForegroundColor Gray
Write-Host "JAVA_HOME: $env:JAVA_HOME" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build sherpa-onnx AAR (Official Method)" -ForegroundColor Cyan
Write-Host "Version: $VERSION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Check prerequisites
Write-Host "`n[1/5] Checking prerequisites..." -ForegroundColor Yellow

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $env:ANDROID_NDK_ROOT)) {
    Write-Host "ERROR: Android NDK not found at $env:ANDROID_NDK_ROOT" -ForegroundColor Red
    Write-Host "  Install NDK via Android Studio: Tools -> SDK Manager -> NDK" -ForegroundColor Yellow
    exit 1
}

Write-Host "  Prerequisites OK" -ForegroundColor Green

# Step 2: Clone sherpa-onnx repository
Write-Host "`n[2/5] Cloning sherpa-onnx ($VERSION)..." -ForegroundColor Yellow

if (Test-Path $SHERPA_DIR) {
    Write-Host "  Removing existing $SHERPA_DIR..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $SHERPA_DIR
}

git clone --depth 1 --branch $VERSION https://github.com/k2-fsa/sherpa-onnx.git $SHERPA_DIR
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to clone sherpa-onnx" -ForegroundColor Red
    exit 1
}
Write-Host "  Cloned to $SHERPA_DIR" -ForegroundColor Green

# Step 3: Download pre-built Android binaries
Write-Host "`n[3/5] Downloading pre-built Android binaries..." -ForegroundColor Yellow

# Remove leading 'v' from version for URL if present
$SHERPA_VERSION = $VERSION -replace '^v', ''
$ANDROID_TAR = "sherpa-onnx-v${SHERPA_VERSION}-android.tar.bz2"
$ANDROID_TAR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_VERSION}/${ANDROID_TAR}"

Write-Host "  Downloading: $ANDROID_TAR_URL" -ForegroundColor Gray

# Use curl.exe (not PowerShell's curl alias) if available, otherwise Invoke-WebRequest
if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    curl.exe -L -o $ANDROID_TAR $ANDROID_TAR_URL
} else {
    Invoke-WebRequest -Uri $ANDROID_TAR_URL -OutFile $ANDROID_TAR
}

Write-Host "  Extracting..." -ForegroundColor Gray

# Extract using 7z if available, otherwise try tar
if (Get-Command "7z" -ErrorAction SilentlyContinue) {
    7z x $ANDROID_TAR -y | Out-Null
} elseif (Get-Command "tar" -ErrorAction SilentlyContinue) {
    tar -xjf $ANDROID_TAR
} else {
    Write-Host "ERROR: No extraction tool found (7z or tar)" -ForegroundColor Red
    exit 1
}

Remove-Item -Force $ANDROID_TAR
Write-Host "  Extracted jniLibs/ structure" -ForegroundColor Green

# Step 4: Copy native libraries to Android module
Write-Host "`n[4/5] Copying native libraries to Android module..." -ForegroundColor Yellow

$SHERPA_ANDROID_DIR = "$SHERPA_DIR\android\SherpaOnnxAar\sherpa_onnx"

# Create jniLibs directories
@("arm64-v8a", "armeabi-v7a", "x86", "x86_64") | ForEach-Object {
    $targetDir = "$SHERPA_ANDROID_DIR\src\main\jniLibs\$_"
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    $sourceDir = "jniLibs\$_"
    if (Test-Path $sourceDir) {
        Get-ChildItem $sourceDir | Copy-Item -Destination $targetDir
        Write-Host "  Copied $_ libraries" -ForegroundColor Gray
    }
}

# Fix broken symlinks - copy actual Kotlin files from kotlin-api to Android module
Write-Host "`n  Fixing broken symlinks in Android module..." -ForegroundColor Yellow

$KOTLIN_API_DIR = "$SHERPA_DIR\sherpa-onnx\kotlin-api"
$ANDROID_JAVA_DIR = "$SHERPA_ANDROID_DIR\src\main\java\com\k2fsa\sherpa\onnx"

if (Test-Path $KOTLIN_API_DIR) {
    Get-ChildItem "$KOTLIN_API_DIR\*.kt" | ForEach-Object {
        Copy-Item $_.FullName "$ANDROID_JAVA_DIR\$($_.Name)"
        Write-Host "  Fixed: $($_.Name)" -ForegroundColor Gray
    }
}

Write-Host "  Symlink fix complete" -ForegroundColor Green

# Step 5: Download Whisper model files
Write-Host "`n[5/5] Downloading Whisper base.en model..." -ForegroundColor Yellow

$MODEL_DIR = "$SHERPA_ANDROID_DIR\src\main\assets\sherpa-onnx-whisper-base.en"
New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null

$MODEL_TAR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2"
$MODEL_TAR = "sherpa-onnx-whisper-base.en.tar.bz2"

Write-Host "  Downloading model..." -ForegroundColor Gray

if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    curl.exe -L -o $MODEL_TAR $MODEL_TAR_URL
} else {
    Invoke-WebRequest -Uri $MODEL_TAR_URL -OutFile $MODEL_TAR
}

Write-Host "  Extracting model files..." -ForegroundColor Gray

# Extract to temp dir first
$TEMP_DIR = [System.IO.Path]::GetTempPath() + [System.IO.Path]::GetRandomFileName()
New-Item -ItemType Directory -Path $TEMP_DIR -Force | Out-Null

if (Get-Command "7z" -ErrorAction SilentlyContinue) {
    7z x "$MODEL_TAR" -o"$TEMP_DIR" -y | Out-Null
    # Extract the .bz2 archive
    $bz2File = Get-ChildItem $TEMP_DIR -Filter "*.tar" | Select-Object -First 1
    if ($bz2File) {
        7z x "$TEMP_DIR\$($bz2File.Name)" -o"$TEMP_DIR" -y | Out-Null
        Remove-Item "$TEMP_DIR\$($bz2File.Name)"
    }
} elseif (Get-Command "tar" -ErrorAction SilentlyContinue) {
    tar -xjf $MODEL_TAR -C $TEMP_DIR
}

# Move model files to destination
$sourceModelDir = Get-ChildItem $TEMP_DIR -Directory | Where-Object { $_.Name -like "*whisper*" } | Select-Object -First 1
if ($sourceModelDir) {
    Get-ChildItem $sourceModelDir.FullName | Copy-Item -Destination $MODEL_DIR -Recurse -Force
} else {
    # Fallback: copy everything
    Get-ChildItem $TEMP_DIR | Copy-Item -Destination $MODEL_DIR -Recurse -Force
}

Remove-Item -Recurse -Force $TEMP_DIR
Remove-Item -Force $MODEL_TAR

# Copy tokens.txt to assets root
if (Test-Path "$MODEL_DIR\base.en-tokens.txt") {
    Copy-Item "$MODEL_DIR\base.en-tokens.txt" "$SHERPA_ANDROID_DIR\src\main\assets\tokens.txt"
}

Write-Host "  Model files ready in $MODEL_DIR" -ForegroundColor Green

# Build the AAR
Write-Host "`nBuilding AAR with Gradle..." -ForegroundColor Yellow

# Use absolute path for Android module (gradlew.bat is in SherpaOnnxAar, not sherpa_onnx)
$SHERPA_ANDROID_DIR_FULL = Join-Path $PWD "$SHERPA_ANDROID_DIR"
$GRADLE_DIR = Split-Path $SHERPA_ANDROID_DIR_FULL -Parent

# Convert paths to forward slashes (Gradle accepts this on Windows)
$WINDOWS_SDK_ROOT = $ANDROID_SDK_ROOT -replace '\\', '/'
$WINDOWS_NDK_ROOT = $env:ANDROID_NDK_ROOT -replace '\\', '/'
Set-Content -Path "$GRADLE_DIR\local.properties" -Value "sdk.dir=$WINDOWS_SDK_ROOT"
Write-Host "  SDK: $WINDOWS_SDK_ROOT" -ForegroundColor Gray
Write-Host "  NDK: $WINDOWS_NDK_ROOT" -ForegroundColor Gray

$GRADLE_PROPERTIES = "$GRADLE_DIR\gradle.properties"
if (Test-Path $GRADLE_PROPERTIES) {
    if (-not (Select-String -Path $GRADLE_PROPERTIES -Pattern "org.gradle.jvmargs" -Quiet)) {
        Add-Content -Path $GRADLE_PROPERTIES ""
        Add-Content -Path $GRADLE_PROPERTIES "org.gradle.jvmargs=-Dfile.encoding=UTF-8"
    }
}

# Set environment variables for Gradle
$env:ANDROID_SDK_ROOT = $WINDOWS_SDK_ROOT
$env:ANDROID_NDK = $WINDOWS_NDK_ROOT

# Clean and build using gradlew.bat (Windows batch file works better in MSYS2)
Write-Host "  Cleaning previous builds..." -ForegroundColor Gray
Push-Location $GRADLE_DIR
.\gradlew.bat clean
if ($LASTEXITCODE -ne 0) { Write-Host "  Clean completed with warnings" -ForegroundColor Yellow }

Write-Host "  Building release AAR..." -ForegroundColor Gray
.\gradlew.bat :sherpa_onnx:assembleRelease -PANDROID_NDK="$WINDOWS_NDK_ROOT"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: AAR build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location

# Copy AAR to VLM-Analyze-Android libs
Write-Host "`nInstalling AAR to VLM-Analyze-Android..." -ForegroundColor Yellow

$AAR_SOURCE = "$SHERPA_ANDROID_DIR_FULL\build\outputs\aar\sherpa_onnx-release.aar"

if (-not (Test-Path $AAR_SOURCE)) {
    Write-Host "ERROR: AAR not found at $AAR_SOURCE" -ForegroundColor Red
    Write-Host "  Check if the build completed successfully" -ForegroundColor Yellow
    exit 1
}
$LIBS_DIR = "libs"
$AAR_DEST = "$LIBS_DIR\sherpa-onnx.aar"

if (-not (Test-Path $LIBS_DIR)) {
    New-Item -ItemType Directory -Path $LIBS_DIR -Force | Out-Null
}

Copy-Item $AAR_SOURCE $AAR_DEST -Force

Write-Host "  AAR copied to: $AAR_DEST" -ForegroundColor Green
Write-Host "  AAR size: $([math]::Round((Get-Item $AAR_DEST).Length / 1MB, 2)) MB" -ForegroundColor Gray

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ready for VLM-Analyze-Android:" -ForegroundColor Yellow
Write-Host "  - $AAR_DEST" -ForegroundColor White
Write-Host "  - $MODEL_DIR" -ForegroundColor White
Write-Host ""
Write-Host "Build APK: .\build.sh -install" -ForegroundColor White
Write-Host ""
