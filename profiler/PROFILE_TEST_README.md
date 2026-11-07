# mtmd-profile-test

Profiling test executable for Qwen2-VL-2B and Qwen3.8-2B performance analysis on Android.

## Purpose

This tool runs multimodal inference with configurable settings and outputs profiling data for performance analysis across different backends (CPU, GPU, NPU).

## Build

### Prerequisites

- MSYS2 UCRT64 environment
- Android NDK 27.3.13750724 (set via `ANDROID_NDK_HOME` in `.env`)
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

| Config File                        | Backend   | Model             | Encoder |
| ---------------------------------- | --------- | ----------------- | ------- |
| `config_htp_1core_vit_q4km.json`   | NPU-1Core | Qwen2-VL Q4_K_M   | HTP0    |
| `config_htp_1core_vit_qwen38.json` | NPU-1Core | Qwen3.8 Q4_K_M    | HTP0    |
| `config_htp_1core_vit.json`        | NPU-1Core | Qwen2-VL Q4_0     | HTP0    |
| `config_htp_1core.json`            | NPU-1Core | Qwen2-VL Q4_0     | Default |
| `config_htp_4core.json`            | NPU-4Core | Qwen2-VL Q4_0     | Default |
| `config_gpu.json`                  | GPU       | Qwen2-VL Q4_K_M   | Default |
| `config_gpu_qwen38.json`           | GPU       | Qwen3.8 Q4_K_M    | Default |
| `config_cpu.json`                  | CPU       | Qwen2-VL Q4_K_M   | Default |
| `config_cpu_qwen38.json`           | CPU       | Qwen3.8 Q4_K_M    | Default |


### Settings Reference

```json
{
  "model_path": "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B-Bartowski/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
  "mmproj_path": "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B-Bartowski/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
  "backend": "NPU-1Core",
  "vit_device": "HTP0",
  "htp_sessions": 1,
  "n_ctx": 16384,
  "n_batch": 512,
  "n_predict": 512,
  "image_min_tokens": 1000,
  "top_k": 40,
  "top_p": 0.9,
  "temperature": 0.9,
  "repetition_penalty": 1.0,
  "image_dir": "/data/local/tmp/mtmd-profile-test/test_images/",
  "htp_profile": 1,
  "system_prompt": "...",
  "user_prompt": "...",
  "output_csv": "/sdcard/profile_results.csv",
  "output_log": "/sdcard/profile_log.txt"
}
```

`vit_device` sets the vision encoder device: `"GPU0"`, `"HTP0"` or `"CPU"`. Leave it out for the default placement.

## Run on Device

### Quick Run (PowerShell)

```powershell
# Clean timing run: per-op, debug and verbose logging stay off
.\run-profile-test.ps1 -Config configs/config_htp_1core_vit_q4km.json -HtpProfile 0

# Run with per-op HTP profiling and reports. Do not use these timings as benchmarks
.\run-profile-test.ps1 -Config configs/config_htp_1core.json -HtpProfile 1
```

This script:
1. Loads environment from `.env` (ADB path, NDK path)
2. Pushes executable, config, and test images to device
3. Pushes required shared libraries
4. Runs at the `-HtpProfile` level, or `htp_profile` from the config if it is not set
5. Pulls results and generates reports automatically

**Output files in `profiler/`:**
- `profile_results*.csv` - Per-image results, named by `output_csv` in the config
- `profile_log.txt` - Raw logcat output
- `thermal_report.txt` - Per-image timing, temperatures and clocks
- `profile_log_utf8.txt` - UTF-8 encoded log (HTP profiling only)
- `htp_official_profile_report.txt` - Official ggml-hexagon-profile report (HTP profiling only)
- `htp_block_profile_report.txt` - Block-grouped analysis report (HTP profiling only)
- `cl_profiling.csv` - OpenCL GPU profiling data (if enabled)
- `cl_profile_report.txt` - OpenCL GPU profile report (if enabled)

### Multiple Runs (Git Bash)

```bash
./run_profiles.sh GPU-Q4KM-Clean=config_gpu.json CPU-Q4KM-Clean=config_cpu.json
```

`run_profiles.sh` takes one `folder=config` pair per run, and runs them in turn with `-HtpProfile 0`:
1. Waits until the GPU and NPU are below 44°C and the CPU is below 47°C, for up to 15 minutes
2. Creates the result folder and copies the config into it
3. Runs `run-profile-test.ps1` and saves its output to `console.txt`
4. Moves the results and reports into the folder

A run is skipped if its folder already exists, so earlier results are not overwritten.

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
    ./mtmd-profile-test --config profile-settings.json --htp-profile 1"
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

### Timing and Thermal Report

```bash
python parse_profile.py profile_log.txt thermal_report.txt --mode thermal
```

Works on clean runs too, because timing and temperatures come from info logs.

### Official HTP Report

```bash
python ../llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py profile_log_utf8.txt -n 100 --sort max-usec
```

## HTP Profiling Levels

| Level | Description                       |
| ----- | --------------------------------- |
| 0     | No profiling                      |
| 1     | Basic per-operation timing        |
| 2     | Extended profiling (more details) |

Set with `-HtpProfile`, `--htp-profile` or `htp_profile` in the config. `mtmd-profile-test` sets `GGML_HEXAGON_PROFILE` from that value before the Hexagon backend loads.

## Backend Selection

| Backend   | Device String         | HTP Sessions |
| --------- | --------------------- | ------------ |
| CPU       | `CPU`                 | N/A          |
| GPU       | `GPU0`                | N/A          |
| NPU-4Core | `HTP0,HTP1,HTP2,HTP3` | 4            |
| NPU-1Core | `HTP0`                | 1            |


The number of HTP sessions is determined by the device string. Sessions are virtual on one cDSP, and 1 session ran faster than 4 on the Fold 7.

## Output Format

### CSV Output

```csv
Image,Backend,Model,PromptTokens,CompletionTokens,TotalTime_ms,TokensPerSecond
VLM_CAMERA_1782100068514.png,NPU-1Core,Qwen3.8-2B-Distill-heretic.Q4_K_M.gguf,90,124,4334.00,33.19
```

`PromptTokens` excludes image tokens, `TotalTime_ms` includes prefill, and `TokensPerSecond` is decode speed.

### HTP Log Output

```
ggml-hex: HTP0 matmul : blk.0.attn_q.weight -> attn_q-0 : 8.234 ms
ggml-hex: HTP0 matmul : blk.0.attn_k.weight -> attn_k-0 : 4.123 ms
ggml-hex: HTP0 matmul : blk.0.ffn_up.weight -> ffn_up-0 : 15.234 ms
```

## Results

Fold 7, 10 test images, Q4_K_M models, clean runs (`-HtpProfile 0`), llama.cpp b12181, Hexagon SDK 6.7.0.1:

| Result Folder                | Model    | Backend | Decode    | Prefill Mean (Range)  | Peak Temp   |
| ---------------------------- | -------- | ------- | --------- | --------------------- | ----------- |
| `HTP-1Core-ViT-Q4KM-Clean`   | Qwen2-VL | NPU     | 40.9 t/s  | 0.72 s (0.50–0.96)    | NPU 89.9°C  |
| `GPU-Q4KM-Clean`             | Qwen2-VL | GPU     | 29.7 t/s  | 5.99 s (1.63–8.91)    | GPU 91.9°C  |
| `CPU-Q4KM-Clean`             | Qwen2-VL | CPU     | 11.5 t/s  | 33.46 s (10.19–43.55) | CPU 105.4°C |
| `HTP-1Core-ViT-Qwen38-Clean` | Qwen3.8  | NPU     | 30.9 t/s  | 0.50 s (0.34–0.60)    | NPU 89.2°C  |
| `GPU-Qwen38-Clean`           | Qwen3.8  | GPU     | 24.7 t/s  | 3.20 s (0.85–5.44)    | GPU 83.0°C  |
| `CPU-Qwen38-Clean`           | Qwen3.8  | CPU     | 10.4 t/s  | 15.42 s (4.23–20.83)  | CPU 105.4°C |

Decode is the mean across images. Answer lengths differ between runs, and Qwen3.8 decode includes thinking tokens.

## Environment Setup

The `run-profile-test.ps1` script automatically loads environment variables from `.env` in the parent directory:

```
ANDROID_SDK_ROOT=/c/Users/misar/AppData/Local/Android/Sdk
ANDROID_NDK_HOME=/c/Users/misar/AppData/Local/Android/Sdk/ndk/27.3.13750724
MSYS2_PATH=/c/msys64
```

Paths are derived from these variables:
- ADB: `$ANDROID_SDK_ROOT/platform-tools/adb.exe`
- libomp.so: `$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/windows-x86_64/lib*/clang/*/lib/linux/aarch64/libomp.so`
