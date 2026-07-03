# mtmd-profile-test

Profiling test executable for Qwen2-VL-2B performance analysis on Android.

## Purpose

This tool runs multimodal inference with configurable settings and outputs profiling data for performance analysis across different backends (CPU, GPU, NPU).

## Build

### Prerequisites

- MSYS2 UCRT64 environment
- Android NDK 25.1.8937393 (set via `ANDROID_NDK_HOME` in `.env`)
- llama.cpp 

### Build Command

```bash
# Build profile test with profiling support (llama.cpp must already be built with -Profile)
./build-mtmd-inference-api.sh -Profile

# Build everything from scratch with profiling
./build-mtmd-inference-api.sh -All -Profile
```

**IMPORTANT: OpenCL GPU profiling requires building with `-Profile` switch**

```bash
# If llama.cpp was NOT built with profiling, rebuild from scratch:
./build-mtmd-inference-api.sh -All -Profile

# If llama.cpp was already built without profiling, this will auto-rebuild:
./build-mtmd-inference-api.sh -Profile

# Check current build configuration:
# Look for GPU_PROFILE=ON in llama.cpp/build-android/.build_config
```

**Output:** `llama.cpp/build-android/bin/mtmd-profile-test`

### Building APK (Profile Switch OFF)

**IMPORTANT: Disable profiling when building the release APK**

```bash
# Build for production/release (no profiling overhead)
./build-mtmd-inference-api.sh

# Or rebuild from scratch without profiling
./build-mtmd-inference-api.sh -All
```

The `-Profile` switch adds OpenCL GPU profiling code which introduces signigicant overhead. Always build without `-Profile` for production APKs.

## Configuration

Config files are in `configs/`:

| Config File             | Backend    | Description                 |
|-------------------------|------------|-----------------------------|
| `config_htp_4core.json` | NPU-4Core  | All 4 HTP cores (HTP0-HTP3) |
| `config_htp_1core.json` | NPU-1Core  | Single HTP core (HTP0 only) |
| `config_gpu.json`       | GPU        | OpenCL (Adreno GPU)         |
| `config_cpu.json`       | CPU        | CPU only                    |


### Settings Reference

```json
{
  "model_path": "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B/Qwen2-VL-2B-Q4_0.gguf",
  "mmproj_path": "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B/mmproj-f16.gguf",
  "backend": "NPU-4Core",
  "htp_sessions": 4,
  "n_ctx": 16384,
  "n_batch": 512,
  "n_predict": 512,
  "image_min_tokens": 1000,
  "top_k": 40,
  "top_p": 0.9,
  "temperature": 0.9,
  "repetition_penalty": 1.0,
  "image_dir": "/data/local/tmp/test_images/",
  "htp_profile": 1,
  "system_prompt": "...",
  "user_prompt": "...",
  "output_csv": "/sdcard/profile_results.csv",
  "output_log": "/sdcard/profile_log.txt"
}
```

## Run on Device

### Quick Run (PowerShell)

```powershell
# Run full profile test with automatic deploy, run, and report generation
.\run-profile-test.ps1 -Config configs/config_htp_1core.json -HtpProfile 1
```

This script:
1. Loads environment from `.env` (ADB path, NDK path)
2. Pushes executable, config, and test images to device
3. Pushes required shared libraries
4. Runs with HTP profiling enabled
5. Pulls results and generates reports automatically

**Output files in `profiler/`:**
- `profile_log.txt` - Raw logcat output
- `profile_log_utf8.txt` - UTF-8 encoded log
- `htp_official_profile_report.txt` - Official ggml-hexagon-profile report
- `htp_block_profile_report.txt` - Block-grouped analysis report
- `cl_profiling.csv` - OpenCL GPU profiling data (if enabled)
- `cl_profile_report.txt` - OpenCL GPU profile report (if enabled)

### Manual Deploy

```bash
# Push executable
adb push llama.cpp/build-android/bin/mtmd-profile-test /data/local/tmp/

# Push config
adb push configs/config_htp_4core.json /data/local/tmp/profile-settings.json

# Push test images (create directory first if needed)
adb shell "mkdir -p /data/local/tmp/test_images"
adb push test_images/ /data/local/tmp/test_images/
```

### Manual Run

```bash
# Run with HTP profiling enabled
adb shell "cd /data/local/tmp && \
    export GGML_HEXAGON_PROFILE=1 && \
    ./mtmd-profile-test --config profile-settings.json"
```

### Pull Results

```bash
adb pull /sdcard/profile_results.csv
adb pull /sdcard/profile_log.txt
```

## Parse Results

### Block-Level Aggregation

```bash
# Parse HTP profile log
python parse_profile.py profile_log.txt htp_block_report.txt

# Parse OpenCL profiling CSV
python parse_profile.py cl_profiling.csv cl_report.txt --mode opencl
```

Parses HTP profiling output and aggregates timing by block:
- `blk.X.attn_q`, `blk.X.attn_k`, `blk.X.attn_v` - Attention weights
- `blk.X.ffn_up`, `blk.X.ffn_gate`, `blk.X.ffn_down` - FFN weights
- `blk.X.norm`, `blk.X.rope` - Normalization and RoPE

### Official HTP Report

```bash
python ../llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py profile_log_utf8.txt -n 100 --sort max-usec
```

## HTP Profiling Levels

| Level | Description                       |
|-------|-----------------------------------|
| 0     | No profiling                      |
| 1     | Basic per-operation timing        |
| 2     | Extended profiling (more details) |

Set via environment variable: `export GGML_HEXAGON_PROFILE=1`

## Backend Selection

| Backend    | Device String            | HTP Sessions |
|------------|--------------------------|--------------|
| CPU        | `CPU`                    | N/A          |
| GPU        | `GPU0`                   | N/A          |
| NPU-4Core  | `HTP0,HTP1,HTP2,HTP3`    | 4            |
| NPU-1Core  | `HTP0`                   | 1            |


The number of HTP sessions is determined by the device string (number of HTP cores specified).

## Output Format

### CSV Output

```csv
Image,Backend,Model,PromptTokens,CompletionTokens,TotalTime_ms,TokensPerSecond
screen_001.png,NPU-4Core,Qwen2-VL-2B-Q4_0,1234,512,5678.9,90.1
```

### HTP Log Output

```
ggml-hex: HTP0 matmul : blk.0.attn_q.weight -> attn_q-0 : 8.234 ms
ggml-hex: HTP0 matmul : blk.0.attn_k.weight -> attn_k-0 : 4.123 ms
ggml-hex: HTP0 matmul : blk.0.ffn_up.weight -> ffn_up-0 : 15.234 ms
```

## Environment Setup

The `run-profile-test.ps1` script automatically loads environment variables from `.env` in the parent directory:

```
ANDROID_SDK_ROOT=/c/Users/misar/AppData/Local/Android/Sdk
ANDROID_NDK_HOME=/c/Users/misar/AppData/Local/Android/Sdk/ndk/25.1.8937393
MSYS2_PATH=/c/msys64
```

Paths are derived from these variables:
- ADB: `$ANDROID_SDK_ROOT/platform-tools/adb.exe`
- libomp.so: `$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/windows-x86_64/lib64/clang/14.0.6/lib/linux/aarch64/libomp.so`
