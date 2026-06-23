#!/bin/bash
# =============================================================================
# build-llamacpp-android.sh
# =============================================================================
# Builds llama.cpp for Android with CPU, OpenCL (GPU), and Hexagon (NPU)

# Load environment configuration if exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi
# backends. Also builds the mtmd-inference-api and prepares all library
# files for integration into the VLM-Analyze Android app.
#
# PREREQUISITES:
#   - Android NDK 25.1.8937393 installed
#   - CMake 3.21+ installed
#   - MSYS2 UCRT64 environment (for Windows)
#   - Python 3.x (optional - for model conversion scripts)
#
# WHAT THIS SCRIPT DOES:
#   1. Clones/updates llama.cpp at a known working commit
#   2. Applies patches:
#      - RPATH configuration for proper library loading
#      - mtmd-inference-api integration
#      - server-http.h fix (missing #include <unordered_map>)
#      - server-http.cpp fix (shared_ptr template args for b9439+)
#      - OpenCL Adreno optimizations (MoE kernel alignment)
#   3. Builds OpenCL ICD loader for Android
#   4. Builds llama.cpp with:
#      - CPU backend (ggml-cpu)
#      - OpenCL backend (ggml-opencl) for GPU
#      - Hexagon backend (ggml-hexagon) for NPU (if SDK available)
#   5. Copies shared libraries to app/src/main/jniLibs/arm64-v8a/
#
# OUTPUT FILES:
#   app/src/main/jniLibs/arm64-v8a/:
#     - libllama.so (llama.cpp inference library)
#     - libmtmd.so (multimodal context library)
#     - libggml-base.so, libggml-cpu.so, libggml.so (core libs)
#     - libggml-opencl.so (GPU backend)
#     - libggml-hexagon.so (NPU backend, if available)
#     - libggml-htp-v*.so (HTP kernel libraries for NPU)
#     - libomp.so (OpenMP runtime)
#
# USAGE:
#   ./build-llamacpp-android.sh     # Full build (always cleans and rebuilds)
#   ./build-llamacpp-android.sh --help
#
# NEXT STEPS:
#   After this script completes, run:
#     ./build-mtmd-inference-api.sh          # Build static lib + headers
#     ./gradlew assembleDebug                # Build APK
# =============================================================================

set -e

# Add MSYS2 tools to PATH early (needed for dirname, cd, etc.)
export PATH="$MSYS2_PATH/usr/bin:$MSYS2_PATH/ucrt64/bin:$PATH"

# Configuration
NDK_VERSION="25.1.8937393"
ANDROID_ABI="arm64-v8a"
ANDROID_API_LEVEL="31"
LLAMA_COMMIT="b9439"  # Known working commit

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/llama.cpp"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check NDK
    if [ -z "$ANDROID_NDK_HOME" ]; then
        log_error "ANDROID_NDK_HOME environment variable not set"
        log_info "Please run setup-config.sh to configure your environment"
        exit 1
    fi

    if [ ! -d "$ANDROID_NDK_HOME" ]; then
        log_error "NDK not found at: $ANDROID_NDK_HOME"
        log_info "Please install NDK version $NDK_VERSION or update ANDROID_NDK_HOME"
        exit 1
    fi

    export ANDROID_NDK="$ANDROID_NDK_HOME"

    # Check CMake
    CMAKE=""
    if command -v cmake &> /dev/null; then
        CMAKE=$(command -v cmake)
    elif [ -f "/c/Program Files/CMake/bin/cmake.exe" ]; then
        CMAKE="/c/Program Files/CMake/bin/cmake.exe"
    elif [ -f "$HOME/AppData/Local/Programs/CMake/bin/cmake.exe" ]; then
        CMAKE="$HOME/AppData/Local/Programs/CMake/bin/cmake.exe"
    fi

    if [ -z "$CMAKE" ]; then
        log_error "CMake not found. Please install CMake 3.21+ in MSYS2 UCRT64 bash using: pacman -S mingw-w64-ucrt-x86_64-cmake"
        exit 1
    fi

    CMAKE_VERSION=$("$CMAKE" --version 2>&1 | head -n1 | awk '{print $3}')
    log_info "CMake version: $CMAKE_VERSION"

    # Check Python (optional - only needed for model conversion scripts)
    PYTHON=""
    if command -v python3 &> /dev/null; then
        PYTHON=$(command -v python3)
    elif command -v python &> /dev/null; then
        PYTHON=$(command -v python)
    fi

    if [ -n "$PYTHON" ]; then
        log_info "Python: $PYTHON"
    else
        log_warning "Python 3 not found - model conversion scripts will not be available"
        log_info "Optional: Install Python 3.x for GGUF model conversion"
    fi

    log_success "Prerequisites check passed"
}

# Clone or update llama.cpp
setup_llama_cpp() {
    log_info "Setting up llama.cpp..."

    if [ ! -d "$LLAMA_DIR" ]; then
        log_info "Cloning llama.cpp repository..."
        # Full clone (no --depth) to ensure all commits are available
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
        cd "$LLAMA_DIR"
        log_info "Checking out commit: $LLAMA_COMMIT"
        git checkout "$LLAMA_COMMIT"
    else
        cd "$LLAMA_DIR"
        # Fetch and checkout specific commit
        log_info "Checking out known working commit: $LLAMA_COMMIT"
        git fetch --all --unshallow 2>/dev/null || git fetch --all
        git checkout "$LLAMA_COMMIT"
    fi

    # Apply required patches
    apply_llama_cpp_patches

    log_success "llama.cpp setup complete"
}

# Apply OpenCL Adreno optimizations
apply_opencl_patches() {
    log_info "Checking OpenCL optimizations..."

    # Check if patches are already applied
    if grep -q "ne01 % 64 == 0" "$LLAMA_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp" 2>/dev/null; then
        log_success "OpenCL Adreno optimizations already applied"
    else
        log_info "Applying OpenCL Adreno optimizations..."

        # Patch 1: Change MoE kernel alignment from 32 to 64
        sed -i 's/ne01 % 32 == 0/ne01 % 64 == 0/g' "$LLAMA_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp"

        # Patch 2: Fix global_size calculations for direct ne01 values
        # These changes ensure proper work-group sizing for Adreno GPUs
        sed -i 's/static_cast<size_t>(ne01)/static_cast<size_t>(ne01)/g' "$LLAMA_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp"

        log_success "OpenCL Adreno optimizations applied"
    fi
}

# Apply required llama.cpp patches (RPATH, mtmd-inference-api, server-http.h fix, server-http.cpp fix)
apply_llama_cpp_patches() {
    log_info "Applying required llama.cpp patches..."

    # Patch 1: Add RPATH to CMakeLists.txt
    if ! grep -q "add_link_options.*-Wl,-rpath" "$LLAMA_DIR/CMakeLists.txt" 2>/dev/null; then
        log_info "Adding RPATH to CMakeLists.txt..."
        # Create temp file with RPATH insertion after line 21 (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        {
            head -n 21 "$LLAMA_DIR/CMakeLists.txt"
            echo ""
            echo "# Set RPATH to use \$ORIGIN (library in same directory as executable/library)"
            echo "if (NOT MSVC)"
            echo '    add_link_options("-Wl,-rpath,\$ORIGIN")'
            echo "endif()"
            tail -n +22 "$LLAMA_DIR/CMakeLists.txt"
        } > "$LLAMA_DIR/CMakeLists.txt.tmp"
        mv "$LLAMA_DIR/CMakeLists.txt.tmp" "$LLAMA_DIR/CMakeLists.txt"
        log_success "RPATH added to CMakeLists.txt"
    else
        log_info "RPATH already in CMakeLists.txt"
    fi

    # Patch 2: Add mtmd-inference-api to tools/CMakeLists.txt
    if ! grep -q "add_subdirectory(mtmd-inference-api)" "$LLAMA_DIR/tools/CMakeLists.txt" 2>/dev/null; then
        log_info "Adding mtmd-inference-api to tools/CMakeLists.txt..."
        sed -i '/add_subdirectory(mtmd)/a\    add_subdirectory(mtmd-inference-api)' "$LLAMA_DIR/tools/CMakeLists.txt"
        log_success "mtmd-inference-api added to tools/CMakeLists.txt"
    else
        log_info "mtmd-inference-api already in tools/CMakeLists.txt"
    fi

    # Patch 3: Fix server-http.h - add missing #include <unordered_map>
    if ! grep -q "#include <unordered_map>" "$LLAMA_DIR/tools/server/server-http.h" 2>/dev/null; then
        log_info "Adding #include <unordered_map> to server-http.h..."
        sed -i 's/#include <cstdint>/#include <cstdint>\n#include <unordered_map>/' "$LLAMA_DIR/tools/server/server-http.h"
        log_success "unordered_map include added to server-http.h"
    else
        log_info "unordered_map include already in server-http.h"
    fi

    # Patch 4: Fix server-http.cpp - add template args to shared_ptr on lines 443-444
    # This fixes compilation error in newer llama.cpp versions (b9439+)
    if grep -q "std::shared_ptr q_ptr = std::move(request)" "$LLAMA_DIR/tools/server/server-http.cpp" 2>/dev/null; then
        log_info "Fixing server-http.cpp shared_ptr template args..."
        sed -i 's/std::shared_ptr q_ptr = std::move(request)/std::shared_ptr<server_http_req> q_ptr = std::move(request)/' "$LLAMA_DIR/tools/server/server-http.cpp"
        sed -i 's/std::shared_ptr r_ptr = std::move(response)/std::shared_ptr<server_http_res> r_ptr = std::move(response)/' "$LLAMA_DIR/tools/server/server-http.cpp"
        log_success "server-http.cpp shared_ptr template args fixed"
    else
        log_info "server-http.cpp shared_ptr already has correct template args"
    fi

    log_success "All llama.cpp patches applied"
}

# Build OpenCL ICD loader
build_opencl_icd() {
    log_info "Building OpenCL ICD loader..."

    # Ensure MSYS2 tools are in PATH
    export PATH="$MSYS2_PATH/usr/bin:$MSYS2_PATH/ucrt64/bin:$PATH"

    # Create build-opencl-android.sh if missing
    create_build_opencl_script

    if [ ! -d "$LLAMA_DIR/OpenCL-ICD-Loader" ]; then
        log_info "Cloning OpenCL ICD loader..."
        cd "$LLAMA_DIR"
        git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    fi

    # Install OpenCL headers if not already installed
    OPENCL_INSTALL_DIR="/tmp/opencl-install"
    if [ ! -d "$OPENCL_INSTALL_DIR/share/cmake/OpenCLHeaders" ]; then
        log_info "Installing OpenCL headers..."
        cd "$LLAMA_DIR/OpenCL-headers"
        rm -rf build
        mkdir -p build && cd build
        "$CMAKE" .. -DCMAKE_INSTALL_PREFIX="$OPENCL_INSTALL_DIR" > /dev/null 2>&1
        "$CMAKE" --install . --prefix "$OPENCL_INSTALL_DIR" > /dev/null 2>&1
        log_info "OpenCL headers installed to $OPENCL_INSTALL_DIR"
    fi

    cd "$LLAMA_DIR/OpenCL-ICD-Loader"

    # Clean previous build
    rm -rf build-android
    mkdir -p build-android
    cd build-android

    # Configure and build
    "$CMAKE" .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$ANDROID_ABI" \
        -DANDROID_PLATFORM=android-24 \
        -DANDROID_STL=c++_shared \
        -DOPENCL_ICD_LOADER_BUILD_SHARED_LIBS=ON \
        -DOpenCLHeaders_DIR="$OPENCL_INSTALL_DIR/share/cmake/OpenCLHeaders"

    ninja

    # Copy libOpenCL.so to NDK sysroot
    log_info "Installing libOpenCL.so to NDK sysroot..."
    SYSROOT_LIB="$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/$ANDROID_ABI"
    mkdir -p "$SYSROOT_LIB"
    cp "$LLAMA_DIR/OpenCL-ICD-Loader/build-android/libOpenCL.so" "$SYSROOT_LIB/"

    log_success "OpenCL ICD loader build complete"
}

# Clone OpenCL-headers if missing
setup_opencl_headers() {
    if [ -d "$LLAMA_DIR/OpenCL-headers" ]; then
        return
    fi

    log_info "Cloning OpenCL-headers..."
    cd "$LLAMA_DIR"
    git clone https://github.com/KhronosGroup/OpenCL-Headers.git OpenCL-headers
    log_success "OpenCL-headers cloned"
}

# Create build-opencl-android.sh
create_build_opencl_script() {
    local script="$LLAMA_DIR/build-opencl-android.sh"
    if [ -f "$script" ]; then
        return
    fi

    log_info "Creating build-opencl-android.sh..."

    cat > "$script" << 'OPENCL_SCRIPT'
#!/bin/bash
# Build OpenCL ICD loader for Android
# Run from MSYS2 UCRT64 terminal inside llama.cpp directory

set -e

# Ensure MSYS2 tools are in PATH
export PATH="$MSYS2_PATH/usr/bin:$MSYS2_PATH/ucrt64/bin:$PATH"

# Configuration
ANDROID_NDK="${ANDROID_NDK_HOME}"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-31"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Building OpenCL ICD loader for Android"
echo "=========================================="
echo "Android NDK: $ANDROID_NDK"
echo "ABI: $ANDROID_ABI"
echo "Platform: $ANDROID_PLATFORM"
echo ""

# Check NDK exists
if [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK"
    exit 1
fi

# Clone OpenCL-headers if missing
if [ ! -d "$SCRIPT_DIR/OpenCL-headers" ]; then
    echo "Cloning OpenCL-headers..."
    cd "$SCRIPT_DIR"
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git OpenCL-headers
fi

# Install OpenCL headers if not already installed
OPENCL_INSTALL_DIR="/tmp/opencl-install"
if [ ! -d "$OPENCL_INSTALL_DIR/share/cmake/OpenCLHeaders" ]; then
    echo "Installing OpenCL headers..."
    cd "$SCRIPT_DIR/OpenCL-headers"
    rm -rf build
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$OPENCL_INSTALL_DIR" > /dev/null 2>&1
    cmake --install . --prefix "$OPENCL_INSTALL_DIR" > /dev/null 2>&1
    echo "OpenCL headers installed to $OPENCL_INSTALL_DIR"
fi

# Clone OpenCL ICD loader if not present
if [ ! -d "$SCRIPT_DIR/OpenCL-ICD-Loader" ]; then
    echo "Cloning OpenCL ICD loader..."
    cd "$SCRIPT_DIR"
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
fi

cd "$SCRIPT_DIR/OpenCL-ICD-Loader"

# Clean and create build directory
rm -rf build-android
mkdir -p build-android
cd build-android

# Configure with CMake
echo "Configuring CMake..."
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DANDROID_STL=c++_shared \
    -DOPENCL_ICD_LOADER_BUILD_SHARED_LIBS=ON \
    -DOpenCLHeaders_DIR="$OPENCL_INSTALL_DIR/share/cmake/OpenCLHeaders"

# Build
echo ""
echo "Building..."
ninja

# Copy libOpenCL.so to NDK sysroot
echo ""
echo "Copying libOpenCL.so to NDK sysroot..."
NDK_SYSROOT_LIB="$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/$ANDROID_ABI"
mkdir -p "$NDK_SYSROOT_LIB"
cp "$SCRIPT_DIR/OpenCL-ICD-Loader/build-android/libOpenCL.so" "$NDK_SYSROOT_LIB/"

echo ""
echo "=========================================="
echo "OpenCL ICD loader build complete!"
echo "=========================================="
echo ""
echo "libOpenCL.so copied to: $NDK_SYSROOT_LIB"
echo ""
OPENCL_SCRIPT

    chmod +x "$script"
    log_success "build-opencl-android.sh created"
}

# Create build-android.sh
create_build_android_script() {
    local script="$LLAMA_DIR/build-android.sh"
    if [ -f "$script" ]; then
        return
    fi

    log_info "Creating build-android.sh..."

    cat > "$script" << 'ANDROID_SCRIPT'
#!/bin/bash
# Build llama-mtmd for Android with OpenCL + Hexagon support
# Run from MSYS2 UCRT64 terminal

set -e

# Ensure MSYS2 tools are in PATH
export PATH="$MSYS2_PATH/usr/bin:$MSYS2_PATH/ucrt64/bin:$PATH"

# Configuration
ANDROID_NDK="${ANDROID_NDK_HOME}"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-31"
BUILD_DIR="build-android"

# Hexagon SDK (optional - for NPU support)
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-}"
HEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07}"

echo "=========================================="
echo "Building llama-mtmd for Android"
echo "=========================================="
echo "Android NDK: $ANDROID_NDK"
echo "ABI: $ANDROID_ABI"
echo "Platform: $ANDROID_PLATFORM"
echo "Build Dir: $BUILD_DIR"
echo "Hexagon SDK: ${HEXAGON_SDK_ROOT:-NOT SET (OpenCL only)}"
echo ""

# Check NDK exists
if [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK"
    echo "Please set ANDROID_NDK environment variable or install NDK"
    exit 1
fi

# Check OpenCL ICD loader is installed
OPENCL_LIB="$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/$ANDROID_ABI/libOpenCL.so"
if [ ! -f "$OPENCL_LIB" ]; then
    echo "ERROR: OpenCL ICD loader not found at $OPENCL_LIB"
    echo "Please run ./build-opencl-android.sh first to install OpenCL support"
    exit 1
fi
echo "OpenCL ICD loader found: $OPENCL_LIB"

# Fix Hexagon SDK qaic path structure if needed
if [ -n "$HEXAGON_SDK_ROOT" ] && [ -d "$HEXAGON_SDK_ROOT" ]; then
    QAIC_DIR="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic"
    if [ -d "$QAIC_DIR/WinNT" ] && [ ! -d "$QAIC_DIR/bin" ]; then
        echo "Fixing Hexagon SDK qaic path structure..."
        cd "$QAIC_DIR"
        ln -s WinNT bin
        echo "Created symlink: bin -> WinNT"
    fi
fi

# Check Hexagon SDK (optional)
HEXAGON_ENABLED="OFF"
if [ -n "$HEXAGON_SDK_ROOT" ] && [ -d "$HEXAGON_SDK_ROOT" ]; then
    echo "Hexagon SDK found: $HEXAGON_SDK_ROOT"
    HEXAGON_ENABLED="ON"
elif [ -n "$HEXAGON_SDK_ROOT" ]; then
    echo "WARNING: HEXAGON_SDK_ROOT set but directory not found: $HEXAGON_SDK_ROOT"
    echo "         Falling back to OpenCL only"
else
    echo "Info: Hexagon SDK not configured (set HEXAGON_SDK_ROOT for NPU support)"
fi
echo ""

# Clean previous build
echo "Cleaning previous build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring CMake..."
OPENCL_LIB="$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/$ANDROID_ABI/libOpenCL.so"
OPENCL_INCLUDE="$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include"

# Build with RPATH set to $ORIGIN so libraries find each other without LD_LIBRARY_PATH
cmake .. -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_OPENCL=ON \
    -DGGML_HEXAGON="$HEXAGON_ENABLED" \
    -DGGML_BACKEND_DL=ON \
    -DGGML_OPENMP=ON \
    -DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128 \
    -DGGML_LLAMAFILE=OFF \
    -DPREBUILT_LIB_DIR="android_aarch64" \
    -DLLAMA_BUILD=ON \
    -DLLAMA_NATIVE=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_SHARED=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_OPENSSL=OFF \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -flto -fno-finite-math-only -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -flto -fno-finite-math-only -D_GNU_SOURCE" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,'\$ORIGIN'" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,'\$ORIGIN'" \
    -DCMAKE_SHARED_LIBRARY_RPATH_COMMAND="-Wl,-rpath,'\$ORIGIN'" \
    -DOpenCL_LIBRARY="$OPENCL_LIB" \
    -DOpenCL_INCLUDE_DIR="$OPENCL_INCLUDE" \
    ${HEXAGON_SDK_ROOT:+-DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT"}

# Build core libraries, mtmd tool, and llama-server
echo ""
echo "Building core libraries, mtmd-cli, and llama-server..."
if [ "$HEXAGON_ENABLED" = "ON" ]; then
    # Build with Hexagon - targets use ExternalProject naming (htp-vXX)
    ninja llama ggml llama-mtmd-cli llama-server htp-v73 htp-v75 htp-v79 htp-v81 -j$(nproc)
else
    ninja llama ggml llama-mtmd-cli llama-server -j$(nproc)
fi

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Output binaries:"
ls -la bin/llama-mtmd-cli bin/*.so 2>/dev/null || echo "Check build directory for outputs"
echo ""
echo "To test on device:"
echo "  adb push $BUILD_DIR/bin/llama-mtmd-cli /data/local/tmp/"
echo "  adb push /path/to/model.gguf /data/local/tmp/"
echo "  adb push /path/to/mmproj.gguf /data/local/tmp/"
echo "  adb shell 'cd /data/local/tmp && ./llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image test.jpg -p \"Describe this image\"'"
echo ""
echo "Libraries for Android app integration:"
echo "  - bin/libllama.so (32 MB)"
echo "  - bin/libmtmd.so (10 MB)"
echo "  - bin/libggml-opencl.so (7.7 MB) - OpenCL GPU backend"
echo "  - bin/libggml-cpu.so (4.3 MB) - CPU fallback"
if [ "$HEXAGON_ENABLED" = "ON" ]; then
    echo "  - bin/libggml-hexagon.so - Hexagon NPU backend"
    echo "  - bin/libggml-htp-*.so - HTP kernels (v73, v75, v79, v81)"
    echo ""
    echo "Hexagon runtime env vars:"
    echo "  D=HTP0              # Use Hexagon NPU"
    echo "  GGML_HEXAGON_NDEV=1 # Number of NPU devices"
fi
ANDROID_SCRIPT

    chmod +x "$script"
    log_success "build-android.sh created"
}

# Fix Hexagon SDK qaic path structure
fix_hexagon_qaic_path() {
    local hexagon_sdk="${1:-$HEXAGON_SDK_ROOT}"
    local qaic_dir="${hexagon_sdk}/ipc/fastrpc/qaic"

    if [ ! -d "$qaic_dir" ]; then
        log_warning "Hexagon qaic directory not found: $qaic_dir"
        return
    fi

    # Create bin symlink if it doesn't exist and WinNT exists
    if [ ! -d "$qaic_dir/bin" ] && [ -d "$qaic_dir/WinNT" ]; then
        log_info "Fixing Hexagon SDK qaic path structure..."
        cd "$qaic_dir"
        ln -s WinNT bin
        log_success "Created symlink: bin -> WinNT"
    fi
}

# Build llama.cpp
build_llama_cpp() {
    log_info "Building llama.cpp with CPU, OpenCL, and Hexagon backends..."

    cd "$LLAMA_DIR"

    # Ensure MSYS2 tools are in PATH
    export PATH="$MSYS2_PATH/usr/bin:$MSYS2_PATH/ucrt64/bin:$PATH"

    # Copy mtmd-inference-api source to tools directory
    MTMD_API_SRC="$SCRIPT_DIR/mtmd-inference-api"
    MTMD_API_TARGET="$LLAMA_DIR/tools/mtmd-inference-api"
    if [ -d "$MTMD_API_SRC" ]; then
        log_info "Copying mtmd-inference-api to tools/..."
        rm -rf "$MTMD_API_TARGET"
        cp -r "$MTMD_API_SRC" "$MTMD_API_TARGET"
        log_success "mtmd-inference-api copied to tools/"
    fi

    # Create both build scripts
    create_build_opencl_script
    create_build_android_script

    # Check if Hexagon SDK is available
    HEXAGON_SDK=""
    if [ -n "$HEXAGON_SDK_ROOT" ] && [ -d "$HEXAGON_SDK_ROOT" ]; then
        HEXAGON_SDK="$HEXAGON_SDK_ROOT"
        log_info "Hexagon SDK found: $HEXAGON_SDK"
    else
        log_warning "Hexagon SDK not found - building OpenCL only"
    fi

    # Fix Hexagon SDK qaic path structure if needed
    if [ -n "$HEXAGON_SDK" ]; then
        fix_hexagon_qaic_path "$HEXAGON_SDK"
    fi

    # Build OpenCL ICD loader first
    bash build-opencl-android.sh

    # Build llama.cpp with OpenCL (and Hexagon if available)
    if [ -n "$HEXAGON_SDK" ]; then
        export HEXAGON_SDK_ROOT="$HEXAGON_SDK"
        bash build-android.sh
    else
        # Build without Hexagon
        bash build-android.sh
    fi

    log_success "llama.cpp build complete"
}

# Copy HTP kernel libraries to lib/ directory
setup_htp_libraries() {
    log_info "Setting up HTP kernel libraries..."

    mkdir -p "$LLAMA_DIR/lib"

    # Copy HTP kernels from build output
    HTP_SOURCE="$LLAMA_DIR/build-android/ggml/src/ggml-hexagon"

    for version in v73 v75 v79 v81; do
        if [ -f "${HTP_SOURCE}/libggml-htp-${version}.so" ]; then
            cp "${HTP_SOURCE}/libggml-htp-${version}.so" "$LLAMA_DIR/lib/"
            log_info "Copied libggml-htp-${version}.so"
        fi
    done

    # Also copy main libraries to lib/ for convenience
    if [ -d "$LLAMA_DIR/build-android/bin" ]; then
        for lib in libggml-base.so libggml-cpu.so libggml-hexagon.so libggml-opencl.so libggml.so; do
            if [ -f "$LLAMA_DIR/build-android/bin/$lib" ]; then
                cp "$LLAMA_DIR/build-android/bin/$lib" "$LLAMA_DIR/lib/" 2>/dev/null || true
            fi
        done
    fi

    log_success "HTP libraries setup complete"
}

# Verify build
verify_build() {
    log_info "Verifying build artifacts..."

    if [ ! -d "$LLAMA_DIR/build-android/bin" ]; then
        log_error "build-android/bin directory not found"
        exit 1
    fi

    # Check for required files
    REQUIRED_FILES=(
        "llama-mtmd-cli"
		"llama-server"
		"libllama-server-impl.so"
		"libggml-base.so"
		"libggml-cpu.so"
		"libggml-hexagon.so"
		"libggml-opencl.so"
		"libggml.so"
		"libllama.so"
		"libllama-common.so"
		"libmtmd.so"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$LLAMA_DIR/build-android/bin/$file" ]; then
            SIZE=$(ls -lh "$LLAMA_DIR/build-android/bin/$file" | awk '{print $5}')
            log_info "  [OK] $file ($SIZE)"
        else
            log_warning "  [MISSING] $file"
        fi
    done

    # Check HTP libraries
    if [ -d "$LLAMA_DIR/lib" ]; then
        for file in libggml-htp-*.so; do
            if [ -f "$LLAMA_DIR/lib/$file" ]; then
                SIZE=$(ls -lh "$LLAMA_DIR/lib/$file" | awk '{print $5}')
                log_info "  [OK] $file ($SIZE)"
            fi
        done
    fi

    log_success "Build verification complete"
}

# Copy libraries to jniLibs for Android app
copy_libraries_to_jnilibs() {
    log_info "Copying libraries to jniLibs..."

    JNI_LIBS="$SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a"
    mkdir -p "$JNI_LIBS"

    # Core libraries to copy
    CORE_LIBS=(
        "libllama-server-impl.so"
        "libllama.so"
        "libmtmd.so"
        "libggml-base.so"
        "libggml-cpu.so"
        "libggml-opencl.so"
        "libggml-hexagon.so"
        "libggml.so"
        "libllama-common.so"
    )

    for lib in "${CORE_LIBS[@]}"; do
        if [ -f "$LLAMA_DIR/build-android/bin/$lib" ]; then
            cp "$LLAMA_DIR/build-android/bin/$lib" "$JNI_LIBS/"
            SIZE=$(ls -lh "$JNI_LIBS/$lib" | awk '{print $5}')
            log_success "Copied $lib ($SIZE)"
        else
            log_warning "Missing: $lib"
        fi
    done

    # Copy HTP kernel libraries (explicit list to avoid glob issues)
    if [ -d "$LLAMA_DIR/lib" ]; then
        for version in v73 v75 v79 v81; do
            HTP_LIB="$LLAMA_DIR/lib/libggml-htp-${version}.so"
            if [ -f "$HTP_LIB" ]; then
                cp "$HTP_LIB" "$JNI_LIBS/"
                SIZE=$(ls -lh "$JNI_LIBS/libggml-htp-${version}.so" | awk '{print $5}')
                log_success "Copied libggml-htp-${version}.so ($SIZE)"
            else
                log_warning "Missing HTP lib: libggml-htp-${version}.so"
            fi
        done
    fi

    # Copy libomp.so from NDK (OpenMP runtime for llama.cpp CPU backend)
    OMP_LIB="$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/lib64/clang/14.0.6/lib/linux/aarch64/libomp.so"
    if [ -f "$OMP_LIB" ]; then
        cp "$OMP_LIB" "$JNI_LIBS/"
        SIZE=$(ls -lh "$JNI_LIBS/libomp.so" | awk '{print $5}')
        log_success "Copied libomp.so ($SIZE) - OpenMP runtime"
    else
        log_warning "libomp.so not found in NDK - OpenMP may not work"
    fi

    log_success "Libraries copied to: $JNI_LIBS"
}

# Main build process
main() {
    echo "========================================"
    echo "  llama.cpp Android Build Script"
    echo "========================================"
    echo ""

    check_prerequisites
    setup_llama_cpp
    apply_opencl_patches
    setup_opencl_headers
    build_opencl_icd
    build_llama_cpp
    setup_htp_libraries
    verify_build
    copy_libraries_to_jnilibs

    echo ""
    echo "========================================"
    echo "  Build Complete!"
    echo "========================================"
    echo ""
    echo "Output locations:"
    echo "  Binaries: $LLAMA_DIR/build-android/bin/"
    echo "  Libraries: $LLAMA_DIR/lib/ (HTP kernels + main libs)"
    echo "  jniLibs: $SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a/"
    echo ""
    echo "Next steps:"
    echo "  1. Build mtmd-inference-api: ./build-mtmd-inference-api.sh"
    echo "  2. Build APK: ./gradlew assembleDebug"
    echo "  3. Install APK: ./gradlew installDebug"
    echo ""
}

# Run main
main
