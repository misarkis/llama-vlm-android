# VLM-Analyze Android - Setup Guide

This guide explains how to set up the build environment on a new computer.

**Important:** MSYS2 UCRT64 is **REQUIRED** - all builds must be done in UCRT64 terminal (llama.cpp cannot be built in PowerShell directly).

## Quick Start

### Option A: PowerShell Workflow (Windows)

1. **Clone the repository**
   ```powershell
   git clone <repo-url>
   cd VLM-Analyze-Android
   ```

2. **Run setup script**
   ```powershell
   .\setup-config.ps1
   ```
   - Follow prompts to configure MSYS2, Android SDK, NDK, Java JDK, and Hexagon SDK paths
   - Creates `.env` file automatically

3. **Build and install**
   ```powershell
   .\buildAPK.ps1 -All -install
   ```
   - Automatically builds all dependencies and APK
   - Installs on connected Android device

---

### Option B: MSYS2 UCRT64 Terminal Workflow

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd VLM-Analyze-Android
   ```

2. **Run setup script**
   ```bash
   source ./setup-config.sh
   ```
   - Follow prompts to configure all dependency paths
   - Creates `.env` file automatically

3. **Build and install**
   ```bash
   ./buildAPK.sh -All -install
   ```
   - Automatically builds all dependencies and APK
   - Installs on connected Android device

---

## Prerequisites

### 1. MSYS2 UCRT64 (REQUIRED - Must Have)

**Install MSYS2:**
1. Download from [msys2.org](https://www.msys2.org/)
2. Install to `C:\msys64`
3. Open **UCRT64 Terminal** (Start Menu → MSYS2 UCRT64)
4. Run:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-libcurl mingw-w64-ucrt-x86_64-libpng mingw-w64-ucrt-x86_64-libjpeg-turbo git
   ```

**Note:** All builds must run in UCRT64 terminal. PowerShell can configure paths, but cannot build llama.cpp. The MSYS2 path is stored in the `.env` file and used by all build scripts.

### 2. Android Studio / SDK

**Install Android Studio:**
1. Download from [developer.android.com](https://developer.android.com/studio)
2. Install with default settings
3. Open Android Studio → SDK Manager
4. Install:
   - Android SDK Platform 34
   - Android SDK Build-Tools 34.0.0
   - Android NDK 25.1.8937393
   - Android SDK Command-line Tools

**Default SDK Location:** `C:\Users\<username>\AppData\Local\Android\Sdk`

### 3. Java JDK 17+

**Included with Android Studio** at `$ANDROID_SDK_ROOT/jbr`, or download from:
- [Adoptium](https://adoptium.net/)
- [Oracle JDK](https://www.oracle.com/java/technologies/downloads/)

### 4. Hexagon SDK (Required - NPU acceleration)

**Download from Qualcomm:**
1. Register at [Qualcomm Developer Portal](https://developer.qualcomm.com/)
2. Download Hexagon SDK 6.6.0.0: [Direct Download](https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Windows/6.6.0.0/Hexagon_SDK_WinNT.zip)
3. Extract to `C:\Hexagon_SDK\6.6.0.0`

**Note:** Required for NPU/HTP acceleration on Qualcomm Snapdragon devices. Without this, only CPU and GPU backends will be available.

### 5. OpenCL SDK (Auto-downloaded - No setup required)

The OpenCL ICD loader is **automatically downloaded and built** during the llama.cpp build process. No manual installation needed.

---

## Configuration

### Run Setup Script

The setup script will:
1. Prompt for each dependency path
2. Validate all paths exist
3. Create a `.env` file with your configuration

**Choose one:**
```powershell
# PowerShell
.\setup-config.ps1
```
```bash
# OR MSYS2 UCRT64 terminal
source ./setup-config.sh
```

### Environment Variables

The setup script creates `.env` with these variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MSYS2_PATH` | MSYS2 installation location | `C:\msys64` |
| `ANDROID_SDK_ROOT` | Android SDK location | `%LOCALAPPDATA%\Android\Sdk` |
| `ANDROID_NDK_HOME` | Android NDK location | `$ANDROID_SDK_ROOT/ndk/25.1.8937393` |
| `JAVA_HOME` | Java JDK location | `$ANDROID_SDK_ROOT/jbr` |
| `HEXAGON_SDK_ROOT` | Hexagon SDK location | `C:\Hexagon_SDK\6.6.0.0` |

**Note:** OpenCL ICD loader is auto-downloaded during build - no environment variable needed.

**Note:** The `.env` file uses Unix-style paths (e.g., `/c/msys64`) for bash compatibility. PowerShell scripts automatically convert them when needed. You only need to run `setup-config.ps1` or `source ./setup-config.sh` once.

---

## Building (MSYS2 UCRT64 Only)

### Full Build (All Dependencies + APK)

```bash
./buildAPK.sh -All -install
```

This will:
1. Build llama.cpp (CPU + OpenCL + Hexagon backends)
2. Build mtmd-inference-api
3. Build sherpa-onnx (voice recognition)
4. Build the Android APK
5. Install on connected device

### Build APK Only

If dependencies already exist:

```bash
./buildAPK.sh -install
```

### Release Build

```bash
./buildAPK.sh -release -install
```

---

## Verifying Installation

### Check ADB

```bash
adb devices
```

Should show your device:
```
List of devices attached
R5CX71YJQNZ   device
```

### Launch App

```bash
adb shell am start -n com.misar.vlmanalyze/.MainActivity
```

### Monitor Logs

```bash
adb logcat -c
adb logcat -s VLM-Application:V VLM-MainActivity:V mtmd-inference:V
```

---

## Troubleshooting

### "Android SDK not found"

Run `source ./setup-config.sh` again and enter the correct SDK path.

### "MSYS2 toolchain not found"

Install the required packages:
```bash
pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake
```

### "Hexagon SDK not found"

1. Download from Qualcomm Developer Portal
2. Extract to `C:\Hexagon_SDK\6.6.0.0`
3. Run `source ./setup-config.sh` and enter the path

### ".env file paths incorrect"

1. Delete the existing `.env` file
2. Run `.\setup-config.ps1` (PowerShell) or `source ./setup-config.sh` (MSYS2) again
3. Verify the `.env` file contains Unix-style paths (e.g., `/c/msys64`)

### "Gradle build failed"

1. Close Android Studio if open
2. Kill Java processes (in PowerShell):
   ```powershell
   Get-Process java -ErrorAction SilentlyContinue | Stop-Process -Force
   ```
3. Clean and rebuild:
   ```bash
   ./buildAPK.sh clean
   ./buildAPK.sh -All -install
   ```

### "ADB device not found"

1. Enable USB Debugging on device (Developer Options)
2. Check cable connection
3. Run `adb devices` to verify

---

## File Structure

```
VLM-Analyze-Android/
├── .env                    # Auto-generated config (created by setup-config)
├── setup-config.ps1        # Windows setup script (PowerShell)
├── setup-config.sh         # MSYS2 setup script (Bash)
├── buildAPK.ps1            # Windows build script (PowerShell)
├── buildAPK.sh             # MSYS2 build script (Bash)
├── build-llamacpp-android.sh    # llama.cpp build (Bash only)
├── build-mtmd-inference-api.ps1 # mtmd API build (PowerShell)
├── build-mtmd-inference-api.sh  # mtmd API build (Bash)
├── build-sherpa-from-source.ps1 # sherpa build (PowerShell)
├── build-sherpa-from-source.sh  # sherpa build (Bash)
├── app/
│   ├── src/main/
│   │   ├── java/           # Kotlin source code
│   │   ├── cpp/            # C++ source code
│   │   ├── jniLibs/        # Prebuilt native libraries
│   │   └── assets/         # Model files (sherpa-onnx)
│   └── build/              # Gradle build outputs
├── llama.cpp/              # Cloned during build (llama.cpp source)
├── mtmd-inference-api/     # Multimodal inference API source
└── libs/                   # Built AAR libraries (sherpa-onnx.aar)
```

---

## Model Setup (After First Build)

1. Download model files:
   - `Qwen2-VL-2B-Q4_K_M.gguf`
   - `mmproj-f16.gguf`

2. Push to device:
   ```bash
   adb push Qwen2-VL-2B-Q4_K_M.gguf /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/
   adb push mmproj-f16.gguf /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/
   ```

3. Configure in app settings

---

## Support

For issues, check:
- `adb logcat` for error messages
- Build script output for dependency errors
- Ensure all paths in `.env` are correct
