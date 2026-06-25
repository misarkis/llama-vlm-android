# llama.cpp Android Build Documentation

## Overview

This document describes how to build llama.cpp for Android with support for CPU, OpenCL (GPU), and Hexagon (NPU) backends.

**Key Point:** The build uses proven scripts (`build-android.sh`, `build-opencl-android.sh`) that have been tested and verified on-device.

**Build Output:** The script automatically copies all libraries to `app/src/main/jniLibs/arm64-v8a/` for packaging in the Android APK.

> **Note:** For initial setup, run `setup-config.ps1` (PowerShell) or `source ./setup-config.sh` (MSYS2). This creates a `.env` file with all required paths. See [SETUP.md](SETUP.md) for details.

## Prerequisites

### Required Tools

1. **MSYS2 UCRT64 Environment**
   - Install MSYS2 from https://www.msys2.org/
   - Use the UCRT64 terminal (not MINGW64)
   - Run `pacman -Syu` to update all packages

2. **Android NDK**: Version `25.1.8937393` (critical - other versions may fail)
   - Download from: https://developer.android.com/ndk/downloads
   - Or via Android Studio: SDK Tools → Android NDK

3. **CMake**: Version `3.21+`
   - Download from: https://cmake.org/download/
   - Or via Android Studio: SDK Tools → CMake

4. **Python**: Version `3.x` (Optional)
   - Required for model conversion scripts only
   - Download from: https://www.python.org/downloads/

5. **Ninja**: Build system
   ```bash
   pacman -S ninja
   ```

6. **Hexagon SDK** (Optional - for NPU support)
   - Qualcomm Hexagon SDK 6.6.0.0 (Community Edition)
   - Required for Snapdragon Hexagon NPU support

### Environment Variables

The `.env` file (created by `setup-config`) sets these automatically:

| Variable | Purpose |
|----------|---------|
| `MSYS2_PATH` | MSYS2 installation (e.g., `/c/msys64`) |
| `ANDROID_NDK_HOME` | Android NDK location |
| `ANDROID_SDK_ROOT` | Android SDK location |
| `HEXAGON_SDK_ROOT` | Hexagon SDK location (optional) |

## Automated Build

Run the automated build script from MSYS2 UCRT64 terminal:

```bash
cd VLM-Analyze-Android
bash build-llamacpp-android.sh
```

This script:
1. Clones llama.cpp at known working commit (b9439)
2. Applies OpenCL Adreno optimizations
3. Builds OpenCL ICD loader
4. Builds llama.cpp with CPU, OpenCL, and Hexagon backends
5. Copies HTP kernel libraries to lib/ directory
6. **Copies all libraries to app/src/main/jniLibs/arm64-v8a/ for APK packaging**
7. **Copies libomp.so from NDK for OpenMP runtime support**

## Manual Build Steps

If the automation script fails, follow these manual steps:

### Step 1: Clone llama.cpp

```bash
cd VLM-Analyze-Android
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b9439  # Known working commit
```

### Step 2: Apply OpenCL Adreno Optimizations

The OpenCL kernels need patches for Adreno GPU optimization:

```bash
# Change MoE kernel alignment from 32 to 64
sed -i 's/ne01 % 32 == 0/ne01 % 64 == 0/g' ggml/src/ggml-opencl/ggml-opencl.cpp

# Add bounds checking to OpenCL kernels (in each .cl file)
# Add at the start of kernel functions:
# if (i01 >= ne01) { return; }
```

### Step 3: Build OpenCL ICD Loader

```bash
cd OpenCL-ICD-Loader
mkdir build-android && cd build-android

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DANDROID_STL=c++_shared \
    -DOPENCL_ICD_LOADER_BUILD_SHARED_LIBS=ON

ninja

# Copy to NDK sysroot
cp libOpenCL.so $ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/
```

### Step 4: Build llama.cpp

```bash
cd ..

# With Hexagon SDK
export HEXAGON_SDK_ROOT="/c/Hexagon_SDK/6.6.0.0"
./build-android.sh

# Without Hexagon (OpenCL only)
./build-android.sh
```

The `build-android.sh` script:
- Uses Ninja build system
- Enables OpenCL GPU backend
- Enables Hexagon NPU backend (if SDK available)
- Builds core libraries (`libllama.so`, `libmtmd.so`, backends) for local multimodal inference

## Expected Output

### build-android/bin/

| File | Size | Purpose |
|------|------|---------|
| `libllama.so` | ~36 MB | llama inference library |
| `libmtmd.so` | ~10.6 MB | Multimodal library (used by JNI wrapper) |
| `libllama-common.so` | ~7.2 MB | Common utilities |
| `libggml-opencl.so` | ~8.0 MB | OpenCL GPU backend |
| `libggml-base.so` | ~7.1 MB | Base ggml library |
| `libggml-hexagon.so` | ~5.2 MB | Hexagon NPU backend |
| `libggml-cpu.so` | ~4.5 MB | CPU backend |
| `libggml.so` | ~1.3 MB | ggml main library |

**Note:** `libllama-server-impl.so` is built but deprecated - not used in current app version.

### lib/ (HTP Kernel Libraries)

| File | Purpose |
|------|---------|
| `libggml-htp-v73.so` | HTP kernels for Hexagon v73 |
| `libggml-htp-v75.so` | HTP kernels for Hexagon v75 |
| `libggml-htp-v79.so` | HTP kernels for Hexagon v79 |
| `libggml-htp-v81.so` | HTP kernels for Hexagon v81 |

### app/src/main/jniLibs/arm64-v8a/ (for APK packaging)

All libraries from `build-android/bin/` plus:
- HTP kernel libraries from `lib/`
- `libomp.so` - OpenMP runtime from NDK

**Note:** JNI wrapper (`libmtmd-inference-jni.so`) is built by Gradle during APK build - links against prebuilt libraries in `jniLibs/` for local multimodal inference.

## Backend Verification

### CPU Backend
```bash
adb shell "cd /data/local/tmp/llama.cpp && LD_LIBRARY_PATH=./lib:./bin ./bin/llama-mtmd-cli -m model.gguf --n-gpu-layers 0 -n 10 -p 'Hello'"
```

### OpenCL GPU Backend (Adreno)
```bash
adb shell "cd /data/local/tmp/llama.cpp && LD_LIBRARY_PATH=./lib:./bin ./bin/llama-mtmd-cli -m model.gguf --n-gpu-layers 999 -n 10 -p 'Hello'"
```

### Hexagon NPU Backend
```bash
adb shell "cd /data/local/tmp/llama.cpp && LD_LIBRARY_PATH=./lib:./bin:/vendor/lib64 ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_NDEV=4 ./bin/llama-mtmd-cli --device HTP0,HTP1,HTP2,HTP3 -m model.gguf --n-gpu-layers 999 -n 10 -p 'Hello'"
```

## Android App Integration

The build script automatically copies all libraries to `app/src/main/jniLibs/arm64-v8a/`. The Android Gradle build compiles JNI wrappers that link against the prebuilt llama.cpp libraries.

**JNI Libraries Built by Gradle:**

| Library | Source | Purpose |
|---------|--------|---------|
| `libvlmanalyze.so` | `native-lib.cpp` | Main app native library (screen capture, TTS, remote API) |
| `libmtmd-inference-jni.so` | `mtmd-inference-jni.cpp` | Local multimodal inference (Qwen2-VL) |

**Note:** The app uses remote HTTP API for server-based inference and `libmtmd-inference-jni.so` for local multimodal inference.

**Build APK:**
```bash
./gradlew assembleDebug
./gradlew installDebug
```

**Library Loading:** The app loads libraries via `System.loadLibrary()` in the Kotlin service classes.

## OpenCL Optimizations

The following optimizations were applied for Adreno GPU:

1. **MoE Kernel Alignment**: Changed from 32 to 64-element alignment for better Adreno performance
2. **Global Work Size**: Direct ne01 values instead of padded calculations
3. **Bounds Checking**: Added `if (i01 >= ne01) return;` to prevent out-of-bounds access in kernel functions

These optimizations are in `ggml/src/ggml-opencl/ggml-opencl.cpp` and all `.cl` kernel files.

## Troubleshooting

### Error: "ANDROID_NDK_HOME not set"
- Run `setup-config.ps1` or `source ./setup-config.sh` to create `.env`
- Verify `ANDROID_NDK_HOME` is set in `.env` file
- Check NDK version is `25.1.8937393`

### Error: "CMake configuration failed"
- Ensure CMake version is `3.21+`
- Check toolchain file path is correct
- Use Ninja generator: `-G Ninja`

### Error: "libOpenCL.so not found"
- Build OpenCL ICD loader first
- Verify it was copied to NDK sysroot

### Error: "HTP libraries not found"
- Verify Hexagon SDK is installed
- Check `lib/` directory for HTP kernel libraries

### Error: "Permission denied" on device
- Run `adb shell "rm -rf /data/local/tmp/llama.cpp"` before pushing
- Ensure ADB is running with proper permissions

## Notes

- **Vulkan backend**: NOT included - OpenCL has explicit Adreno optimization
- **Quantization**: Q4_K_M recommended for Hexagon, Q4_0 for OpenCL
- **Memory**: Minimum 4GB RAM recommended for 2B models
- **Device compatibility**: Snapdragon 8xx series recommended for Hexagon support
- **Known working commit**: b9439
- **Setup**: Run `setup-config.ps1` or `source ./setup-config.sh` first - see [SETUP.md](SETUP.md)
- **Local inference**: Uses `libmtmd-inference-jni.so` for multimodal VLM inference (Qwen2-VL)
- **Remote inference**: App connects to remote HTTP API (llama.cpp server)
- **OpenMP runtime**: `libomp.so` is copied from the NDK for CPU backend parallelization support.