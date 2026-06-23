#!/bin/bash
# =============================================================================
# buildAPK.sh
# =============================================================================
# Builds the VLM-Analyze Android APK with all dependencies.
#
# BUILD FLOW:
#   -All:      Build llama.cpp → mtmd-inference-api → build-sherpa-from-source.sh → APK
#   Default:   Build APK only (uses existing libraries if present)
#
# USAGE:
#   ./buildAPK.sh                     # Build APK only (debug, auto-detect deps)
#   ./buildAPK.sh -install            # Build APK + install on device
#   ./buildAPK.sh -All                # Full rebuild: all deps + APK
#   ./buildAPK.sh -All -install       # Full rebuild + install on device
#   ./buildAPK.sh -release            # Build release APK
#   ./buildAPK.sh clean               # Clean Gradle build artifacts
#   ./buildAPK.sh --help              # Show this help
#
# OUTPUT:
#   Debug APK:  app/build/outputs/apk/debug/app-debug.apk
#   Release:    app/build/outputs/apk/release/app-release.apk
# =============================================================================

# Load environment configuration if exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

set -e

# Parse arguments
BUILD_ALL=false
ACTION="debug"
INSTALL=false

for arg in "$@"; do
    case $arg in
        -All) BUILD_ALL=true ;;
        -install) INSTALL=true ;;
        release) ACTION="release" ;;
        clean) ACTION="clean" ;;
        --help|-h)
            echo "Usage: ./buildAPK.sh [options]"
            echo ""
            echo "Options:"
            echo "  -All        Build llama.cpp + mtmd-inference-api + sherpa + APK"
            echo "  -install    Install APK on connected device after build"
            echo "  -release    Build release APK (default: debug)"
            echo "  clean       Clean Gradle build artifacts"
            echo "  --help      Show this help"
            exit 0
            ;;
    esac
done

GRADLE_WRAPPER="./gradlew"

if [ ! -f "$GRADLE_WRAPPER" ]; then
    echo "Gradle wrapper not found. Using system gradle..."
    GRADLE_WRAPPER="gradle"
fi

# Helper functions
has_llama_libs() {
    [ -f "app/src/main/jniLibs/arm64-v8a/libllama.so" ]
}

has_mtmd_api() {
    [ -f "app/src/main/cpp/libs/arm64-v8a/libmtmd-inference-api.a" ]
}

if [ "$ACTION" = "clean" ]; then
    echo "Cleaning VLM-Analyze Android..."
    $GRADLE_WRAPPER clean || {
        echo "Gradle clean failed (files may be locked). Attempting manual cleanup..."
        rm -rf app/build .gradle 2>/dev/null || true
        echo "Manual cleanup complete."
    }
else
    CONFIG=$(echo "$ACTION" | sed 's/\b\(.\)/\u\1/g')
    BUILD_TASK="assemble$CONFIG"

    # Step 1: Build llama.cpp if -All flag is set
    if [ "$BUILD_ALL" = true ]; then
        echo "=============================================="
        echo "Building llama.cpp (CPU + OpenCL + Hexagon)..."
        echo "=============================================="
        ./build-llamacpp-android.sh

        if [ $? -ne 0 ]; then
            echo "ERROR: llama.cpp build failed"
            exit 1
        fi
        echo "llama.cpp build complete!"
    else
        # Auto-detect: build llama.cpp if libraries missing
        if ! has_llama_libs; then
            echo "llama.cpp libraries not found in jniLibs..."
            echo "Running build-llamacpp-android.sh..."
            ./build-llamacpp-android.sh

            if [ $? -ne 0 ]; then
                echo "ERROR: llama.cpp build failed"
                exit 1
            fi
            echo "llama.cpp build complete!"
        fi
    fi

    # Step 2: Build mtmd-inference-api if -All flag is set
    if [ "$BUILD_ALL" = true ]; then
        echo "=============================================="
        echo "Building mtmd-inference-api..."
        echo "=============================================="
        ./build-mtmd-inference-api.sh

        if [ $? -ne 0 ]; then
            echo "ERROR: mtmd-inference-api build failed"
            exit 1
        fi
        echo "mtmd-inference-api build complete!"
    else
        # Auto-detect: build mtmd-inference-api if static lib missing
        if ! has_mtmd_api; then
            echo "mtmd-inference-api not found..."
            echo "Running build-mtmd-inference-api.sh..."
            ./build-mtmd-inference-api.sh

            if [ $? -ne 0 ]; then
                echo "ERROR: mtmd-inference-api build failed"
                exit 1
            fi
            echo "mtmd-inference-api build complete!"
        fi
    fi

    # Step 3: Build sherpa-onnx if -All flag is set
    if [ "$BUILD_ALL" = true ]; then
        echo "=============================================="
        echo "Building sherpa-onnx..."
        echo "=============================================="
        if [ -f "./build-sherpa-from-source.sh" ]; then
            ./build-sherpa-from-source.sh

            if [ $? -ne 0 ]; then
                echo "WARNING: sherpa-onnx build failed (may already be built)"
            else
                echo "sherpa-onnx build complete!"
            fi
        else
            echo "Info: build-sherpa-from-source.sh not found - skipping sherpa build"
        fi
    fi

    echo "=============================================="
    echo "Building VLM-Analyze Android ($CONFIG)..."
    echo "=============================================="
    $GRADLE_WRAPPER $BUILD_TASK

    if [ $? -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "Build successful!"
        echo "=============================================="
        if [ "$ACTION" = "debug" ]; then
            APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
            echo "Debug APK: $APK_PATH"

            if [ "$INSTALL" = true ]; then
                echo "Installing on connected device..."
                $GRADLE_WRAPPER installDebug
            fi
        else
            APK_PATH="app/build/outputs/apk/release/app-release.apk"
            echo "Release APK: $APK_PATH"
            echo ""
            echo "To install manually: adb install $APK_PATH"
        fi
        echo ""
    else
        echo "Build failed (files may be locked by Gradle/Android Studio)."
        echo "Try: pkill -f java && rm -rf app/build .gradle"
        exit 1
    fi
fi
