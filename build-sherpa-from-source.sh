#!/bin/bash
# Build sherpa-onnx AAR for VLM-Analyze Android
# Uses official pre-built binary tarball (simpler approach)
#
# Prerequisites:
#   - MSYS2 UCRT64 environment
#   - Git
#   - Android SDK with NDK (for AAR build step only)
#   - Java JDK 17+
#
# Usage:
#   ./build-sherpa-from-source.sh [version]
#   Example: ./build-sherpa-from-source.sh v1.13.2

# Load environment configuration if exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

set -e

# Parse arguments
VERSION="${1:-v1.13.2}"
SHERPA_DIR="sherpa-onnx"

# Auto-detect Android SDK Root
if [ -z "$ANDROID_SDK_ROOT" ]; then
    if [ -d "$ANDROID_HOME" ]; then
        ANDROID_SDK_ROOT="$ANDROID_HOME"
    fi
fi

# Use ANDROID_NDK_HOME from .env
if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: ANDROID_NDK_HOME not set in .env"
    exit 1
fi
ANDROID_NDK_ROOT="$ANDROID_NDK_HOME"

# Auto-detect JAVA_HOME
if [ -z "$JAVA_HOME" ]; then
    if [ -d "$ANDROID_SDK_ROOT/../jbr" ]; then
        JAVA_HOME="$ANDROID_SDK_ROOT/../jbr"
    fi
fi
export JAVA_HOME

echo "Android SDK: ${ANDROID_SDK_ROOT:-Not set}"
echo "Android NDK: ${ANDROID_NDK_ROOT:-Not found}"
echo "JAVA_HOME: ${JAVA_HOME:-Not set}"
echo "========================================"
echo "Build sherpa-onnx AAR (Official Method)"
echo "Version: $VERSION"
echo "========================================"

# Step 1: Check prerequisites
echo ""
echo "[1/5] Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "ERROR: Git not found"
    exit 1
fi

if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: Android NDK not found"
    echo "  Install NDK via Android Studio: Tools -> SDK Manager -> NDK"
    exit 1
fi

echo "  Prerequisites OK"

# Step 2: Clone sherpa-onnx repository
echo ""
echo "[2/5] Cloning sherpa-onnx ($VERSION)..."

if [ -d "$SHERPA_DIR" ]; then
    echo "  Removing existing $SHERPA_DIR..."
    rm -rf "$SHERPA_DIR"
fi

git clone --depth 1 --branch "$VERSION" https://github.com/k2-fsa/sherpa-onnx.git "$SHERPA_DIR"
echo "  Cloned to $SHERPA_DIR"

# Step 3: Download pre-built Android binaries
echo ""
echo "[3/5] Downloading pre-built Android binaries..."

# Remove leading 'v' from version for URL if present
SHERPA_VERSION="${VERSION#v}"
ANDROID_TAR="sherpa-onnx-v${SHERPA_VERSION}-android.tar.bz2"
ANDROID_TAR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_VERSION}/${ANDROID_TAR}"

echo "  Downloading: $ANDROID_TAR_URL"
curl -L -o "$ANDROID_TAR" "$ANDROID_TAR_URL"

echo "  Extracting..."
tar -xjf "$ANDROID_TAR"
rm -f "$ANDROID_TAR"

echo "  Extracted jniLibs/ structure"

# Step 4: Copy native libraries to Android module
echo ""
echo "[4/5] Copying native libraries to Android module..."

SHERPA_ANDROID_DIR="$SHERPA_DIR/android/SherpaOnnxAar/sherpa_onnx"

# Create jniLibs directories
mkdir -p "$SHERPA_ANDROID_DIR/src/main/jniLibs/arm64-v8a"
mkdir -p "$SHERPA_ANDROID_DIR/src/main/jniLibs/armeabi-v7a"
mkdir -p "$SHERPA_ANDROID_DIR/src/main/jniLibs/x86"
mkdir -p "$SHERPA_ANDROID_DIR/src/main/jniLibs/x86_64"

# Copy libraries for each ABI
if [ -d "jniLibs/arm64-v8a" ]; then
    cp -v "jniLibs/arm64-v8a/"* "$SHERPA_ANDROID_DIR/src/main/jniLibs/arm64-v8a/"
    echo "  Copied arm64-v8a libraries"
fi

if [ -d "jniLibs/armeabi-v7a" ]; then
    cp -v "jniLibs/armeabi-v7a/"* "$SHERPA_ANDROID_DIR/src/main/jniLibs/armeabi-v7a/"
    echo "  Copied armeabi-v7a libraries"
fi

if [ -d "jniLibs/x86" ]; then
    cp -v "jniLibs/x86/"* "$SHERPA_ANDROID_DIR/src/main/jniLibs/x86/"
    echo "  Copied x86 libraries"
fi

if [ -d "jniLibs/x86_64" ]; then
    cp -v "jniLibs/x86_64/"* "$SHERPA_ANDROID_DIR/src/main/jniLibs/x86_64/"
    echo "  Copied x86_64 libraries"
fi

# Fix broken symlinks - copy actual Kotlin files from kotlin-api to Android module
echo ""
echo "  Fixing broken symlinks in Android module..."

# The kotlin-api is inside sherpa-onnx/ subdirectory
KOTLIN_API_DIR="$SHERPA_DIR/sherpa-onnx/kotlin-api"
ANDROID_JAVA_DIR="$SHERPA_ANDROID_DIR/src/main/java/com/k2fsa/sherpa/onnx"

if [ -d "$KOTLIN_API_DIR" ]; then
    # Copy Kotlin API files to replace symlink placeholders
    for file in "$KOTLIN_API_DIR"/*.kt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            cp "$file" "$ANDROID_JAVA_DIR/$filename"
            echo "  Fixed: $filename"
        fi
    done
fi

echo "  Symlink fix complete"

# Step 5: Download Whisper model files
echo ""
echo "[5/5] Downloading Whisper base.en model..."

MODEL_DIR="$SHERPA_ANDROID_DIR/src/main/assets/sherpa-onnx-whisper-base.en"
mkdir -p "$MODEL_DIR"

MODEL_TAR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2"
MODEL_TAR="sherpa-onnx-whisper-base.en.tar.bz2"

echo "  Downloading model..."
curl -L -o "$MODEL_TAR" "$MODEL_TAR_URL"

echo "  Extracting model files..."
TEMP_DIR=$(mktemp -d)
tar -xjf "$MODEL_TAR" -C "$TEMP_DIR"
if [ -d "$TEMP_DIR/sherpa-onnx-whisper-base.en" ]; then
    cp -r "$TEMP_DIR/sherpa-onnx-whisper-base.en/"* "$MODEL_DIR/"
else
    cp -r "$TEMP_DIR"/* "$MODEL_DIR/"
fi
rm -rf "$TEMP_DIR"
rm -f "$MODEL_TAR"

# Copy tokens.txt to assets root
if [ -f "$MODEL_DIR/base.en-tokens.txt" ]; then
    cp "$MODEL_DIR/base.en-tokens.txt" "$SHERPA_DIR/android/SherpaOnnxAar/sherpa_onnx/src/main/assets/tokens.txt"
fi

echo "  Model files ready in $MODEL_DIR"

# Build the AAR
echo ""
echo "Building AAR with Gradle..."

# Use absolute path for Android module
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHERPA_ANDROID_DIR="$SCRIPT_DIR/$SHERPA_DIR/android/SherpaOnnxAar"

# Ensure JAVA_HOME is exported
export JAVA_HOME

# Setup gradle properties
WINDOWS_SDK_ROOT=$(echo "$ANDROID_SDK_ROOT" | sed 's|^/c/|C:/|')
echo "sdk.dir=$WINDOWS_SDK_ROOT" > "$SHERPA_ANDROID_DIR/local.properties"

if ! grep -q "org.gradle.jvmargs" "$SHERPA_ANDROID_DIR/gradle.properties" 2>/dev/null; then
    echo "" >> "$SHERPA_ANDROID_DIR/gradle.properties"
    echo "org.gradle.jvmargs=-Dfile.encoding=UTF-8" >> "$SHERPA_ANDROID_DIR/gradle.properties"
fi

# Clean and build using Windows gradlew.bat to avoid MSYS2 path issues
echo "  Cleaning previous builds..."
( cd "$SHERPA_ANDROID_DIR" && ./gradlew.bat clean --quiet 2>/dev/null ) || true

echo "  Building release AAR..."
( cd "$SHERPA_ANDROID_DIR" && ./gradlew.bat :sherpa_onnx:assembleRelease -PANDROID_NDK="$ANDROID_NDK_ROOT" )

# Copy AAR to VLM-Analyze-Android libs
echo ""
echo "Installing AAR to VLM-Analyze-Android..."

AAR_SOURCE="$SHERPA_DIR/android/SherpaOnnxAar/sherpa_onnx/build/outputs/aar/sherpa_onnx-release.aar"
LIBS_DIR="libs"
AAR_DEST="$LIBS_DIR/sherpa-onnx.aar"

mkdir -p "$LIBS_DIR"
cp "$AAR_SOURCE" "$AAR_DEST"

echo "  AAR copied to: $AAR_DEST"
echo "  AAR size: $(du -h "$AAR_DEST" | cut -f1)"

# Summary
echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Ready for VLM-Analyze-Android:"
echo "  - $AAR_DEST"
echo "  - $MODEL_DIR"
echo ""
echo "Build APK: ./build.sh -install"
echo ""
