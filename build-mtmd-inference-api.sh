#!/bin/bash
# =============================================================================
# build-mtmd-inference-api.sh
# =============================================================================
# Builds the mtmd-inference-api static library for Android and prepares
# all integration files needed for the VLM-Analyze Android app.
#
# PREREQUISITES:
#   - MSYS2 UCRT64 environment with bash
#   - Android NDK 25.1.8937393 (set via ANDROID_NDK_HOME env var)
#   - CMake 3.21+
#
#   Note: For -All, llama.cpp will be built automatically.
#   For normal usage, ensure llama.cpp is already built (run build-llamacpp-android.sh first).
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

# Load environment configuration if exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi
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
#     app/src/main/cpp/libs/arm64-v8a/
#       - libmtmd-inference-api.a (static library for linking)
#
# USAGE:
#   ./build-mtmd-inference-api.sh     # Build mtmd-inference-api only
#   ./build-mtmd-inference-api.sh -All # Build llama.cpp + mtmd-inference-api
#   ./build-mtmd-inference-api.sh --help
#
# NEXT STEP:
#   After this script completes, build the APK:
#     ./gradlew assembleDebug
#     ./gradlew installDebug
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"
MTMD_SOURCE_DIR="${SCRIPT_DIR}/mtmd-inference-api"
MTMD_TARGET_DIR="${LLAMA_CPP_DIR}/tools/mtmd-inference-api"
BUILD_DIR="${LLAMA_CPP_DIR}/build-android"
APP_MTMD_API_DIR="${SCRIPT_DIR}/app/src/main/cpp/mtmd-api"
APP_MTMD_API_INCLUDE_DIR="${APP_MTMD_API_DIR}/include"
APP_LIBS_DIR="${SCRIPT_DIR}/app/src/main/cpp/libs/arm64-v8a"

info() { echo "[INFO] $1"; }
success() { echo "[SUCCESS] $1"; }
warn() { echo "[WARN] $1"; }
error() { echo "[ERROR] $1"; exit 1; }

BUILD_ALL=false
for arg in "$@"; do
    case $arg in
        -All)
            BUILD_ALL=true
            ;;
        -help|--help)
            echo "Usage: ./build-mtmd-inference-api.sh [-All]"
            echo ""
            echo "Options:"
            echo "  -All  Build llama.cpp + mtmd-inference-api"
            exit 0
            ;;
    esac
done

info "Checking mtmd-inference-api source directory..."
if [ ! -d "$MTMD_SOURCE_DIR" ]; then
    error "mtmd-inference-api source not found at $MTMD_SOURCE_DIR"
fi
success "Source directory found"

info "Copying mtmd-inference-api to llama.cpp/tools/..."
if [ -d "$MTMD_TARGET_DIR" ]; then
    rm -rf "$MTMD_TARGET_DIR"
fi
mkdir -p "$MTMD_TARGET_DIR"
cp -r "${MTMD_SOURCE_DIR}"/* "$MTMD_TARGET_DIR/"
success "Copied to $MTMD_TARGET_DIR"

info "Checking llama.cpp integration..."
TOOLS_CMAKE="${LLAMA_CPP_DIR}/tools/CMakeLists.txt"
if [ -f "$TOOLS_CMAKE" ]; then
    if ! grep -q 'add_subdirectory(mtmd-inference-api)' "$TOOLS_CMAKE"; then
        info "Adding mtmd-inference-api to tools/CMakeLists.txt..."
        sed -i '/add_subdirectory(mtmd)/a\    add_subdirectory(mtmd-inference-api)' "$TOOLS_CMAKE"
        success "Updated tools/CMakeLists.txt"
    else
        info "mtmd-inference-api already integrated in CMakeLists.txt"
    fi
else
    warn "tools/CMakeLists.txt not found"
fi

if [ "$BUILD_ALL" = true ]; then
    info "Building llama.cpp + mtmd-inference-api..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        success "Cleaned build directory"
    fi
    info "Rebuilding llama.cpp with Android configuration..."
    BUILD_SCRIPT="${SCRIPT_DIR}/build-llamacpp-android.sh"
    if [ -f "$BUILD_SCRIPT" ]; then
        export ANDROID_NDK="${ANDROID_NDK_HOME}"
        cd "$SCRIPT_DIR"
        bash "$BUILD_SCRIPT"
        if [ $? -ne 0 ]; then
            error "build-llamacpp-android.sh failed"
        fi
        success "llama.cpp rebuilt successfully"
    else
        error "build-llamacpp-android.sh not found at $BUILD_SCRIPT"
    fi
fi

info "Building mtmd-inference-api library..."
cmake --build "$BUILD_DIR" --target mtmd-inference-api mtmd-inference-test -- -j$(nproc)
if [ $? -ne 0 ]; then
    error "Build failed"
fi
success "Build completed!"

STATIC_LIB="${BUILD_DIR}/tools/mtmd-inference-api/libmtmd-inference-api.a"
TEST_EXE="${BUILD_DIR}/bin/mtmd-inference-test"

info ""
info "Build outputs:"
if [ -f "$STATIC_LIB" ]; then
    SIZE=$(du -h "$STATIC_LIB" | cut -f1)
    info "  Static library: $SIZE"
else
    error "Static library not found at $STATIC_LIB"
fi

if [ -f "$TEST_EXE" ]; then
    SIZE=$(du -h "$TEST_EXE" | cut -f1)
    info "  Test executable: $SIZE"
fi

info ""
info "Copying files to app directory..."
if [ -d "$APP_MTMD_API_DIR" ]; then
    rm -rf "$APP_MTMD_API_DIR"
fi
mkdir -p "$APP_MTMD_API_INCLUDE_DIR"
mkdir -p "$APP_LIBS_DIR"

cp "${MTMD_SOURCE_DIR}/mtmd-inference-api.h" "$APP_MTMD_API_DIR/"
info "  Copied mtmd-inference-api.h"

cp "$STATIC_LIB" "$APP_LIBS_DIR/"
info "  Copied libmtmd-inference-api.a"

LLAMA_H="${LLAMA_CPP_DIR}/include/llama.h"
if [ -f "$LLAMA_H" ]; then
    cp "$LLAMA_H" "$APP_MTMD_API_INCLUDE_DIR/"
    info "  Copied llama.h"
else
    warn "llama.h not found at $LLAMA_H"
fi

GGML_INCLUDE_DIR="${LLAMA_CPP_DIR}/ggml/include"
if [ -d "$GGML_INCLUDE_DIR" ]; then
    for header in "$GGML_INCLUDE_DIR"/*.h; do
        if [ -f "$header" ]; then
            cp "$header" "$APP_MTMD_API_INCLUDE_DIR/"
            info "  Copied $(basename "$header")"
        fi
    done
else
    warn "ggml include directory not found at $GGML_INCLUDE_DIR"
fi

success "Files copied to app directory"

info ""
info "========================================"
info "  mtmd-inference-api Build Complete"
info "========================================"
info ""
info "App integration files:"
info "  Headers: $APP_MTMD_API_INCLUDE_DIR"
info "  Static lib: $APP_LIBS_DIR"
info ""
info "Shared libraries (already copied by build-llamacpp-android.sh):"
info "  Location: ${SCRIPT_DIR}/app/src/main/jniLibs/arm64-v8a/"
info ""
info "Next: Build the app with ./gradlew assembleDebug"
info ""
