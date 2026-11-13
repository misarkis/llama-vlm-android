# mtmd-profile-test

Profiling test executable for on-device VLM performance analysis on Android: Qwen2-VL-2B, Qwen3.8-2B, Qwen3.5 (0.8B, 2B, 4B), Qwen3-VL-2B, LFM2.5-VL-1.6B, SmolVLM2-2.2B and Granite Vision 3.2-2B.

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

The `-Profile` switch adds OpenCL GPU profiling code which introduces significant overhead. Always build without `-Profile` for production APKs.

## Configuration

Config files are in `configs/`:

Each candidate has one config per backend, all Q4_K_M:

| Config File                               | Backend   | Encoder |
| ----------------------------------------- | --------- | ------- |
| `config_htp_1core_vit_<model>.json`       | NPU-1Core | HTP0    |
| `config_gpu_<model>.json`                 | GPU       | Default |
| `config_cpu_<model>.json`                 | CPU       | Default |
| `config_flip6_htp_1core_vit_<model>.json` | NPU-1Core | HTP0    |
| `config_flip6_cpu_<model>.json`           | CPU       | Default |

`<model>` is `qwen2vl`, `qwen38`, `qwen35_08b`, `qwen35_2b`, `qwen35_4b`, `qwen3vl`, `lfm25`, `smolvlm` or `granite`. Output files are tagged the same way (`profile_results_gpu_<model>.csv`; `1core_vit_<model>` for NPU). The `config_flip6_` configs are for the Galaxy Z Flip 6: the same settings with a context size of 4096 (`n_ctx`), and no NPU config for `qwen35_4b`, which does not fit on its NPU.

The per-model configs share the same settings: sampling temperature 0.6, top-k 20, top-p 0.95, repetition penalty 1.0 (min_p is fixed at 0.05 in the API); thinking off with Jinja templates; the short-answer system prompt; image token bounds at the model's default (`-1`); `n_predict` 512.

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
  "image_min_tokens": -1,
  "image_max_tokens": -1,
  "top_k": 20,
  "top_p": 0.95,
  "temperature": 0.6,
  "repetition_penalty": 1.0,
  "image_dir": "/data/local/tmp/mtmd-profile-test/test_images/",
  "htp_profile": 1,
  "system_prompt": "...",
  "user_prompt": "...",
  "output_csv": "/sdcard/profile_results.csv",
  "output_log": "/sdcard/profile_log.txt"
}
```

`vit_device` sets the vision encoder device: `"GPUOpenCL"`, `"HTP0"` or `"CPU"`. Leave it out for the default placement.

`image_min_tokens` / `image_max_tokens`: `-1` keeps the model's own image preprocessing, as accuracy leaderboards do. A fixed budget (min = max = N) is honoured by the Qwen family, partly by LFM2.5 and SmolVLM2, and not by Granite (fixed tile grid).

## Run on Device

### Quick Run (PowerShell)

```powershell
# Clean timing run: per-op, debug and verbose logging stay off
.\run-profile-test.ps1 -Config configs/config_htp_1core_vit_qwen2vl.json -HtpProfile 0

# Run with per-op HTP profiling and reports. Do not use these timings as benchmarks
.\run-profile-test.ps1 -Config configs/config_htp_1core_vit_qwen2vl.json -HtpProfile 1
```

This script:
1. Loads environment from `.env` (ADB path, NDK path)
2. Pushes executable, config, and test images to device
3. Pushes required shared libraries
4. Runs at the `-HtpProfile` level, or `htp_profile` from the config if it is not set
5. Pulls results and generates reports automatically

**Output files in `profiler/`** (`run_profiles.sh` moves them into `Results/<folder>/`):
- `profile_results*.csv` - Per-image results, named by `output_csv` in the config
- `profile_log.txt` - Raw logcat output
- `thermal_report.txt` - Per-image timing, temperatures and clocks
- `profile_log_utf8.txt` - UTF-8 encoded log (HTP profiling only)
- `htp_official_profile_report.txt` - Official ggml-hexagon-profile report (HTP profiling only)
- `htp_block_profile_report.txt` - Block-grouped analysis report (HTP profiling only)
- `cl_profiling.csv` - OpenCL GPU profiling data (if enabled)
- `cl_profile_report.txt` - OpenCL GPU profile report (if enabled)

`console.txt` (written by `run_profiles.sh`) holds mtmd's stderr: image encode time per slice (`image slice encoded in N ms`), image token count (`n_tokens_batch = N`) and image-token decode time, split per image by `=== Processing image i/N` markers.

### Profiling Round (Bash)

```bash
cd profiler
# Clean timing (HTP_PROFILE defaults to 0); one folder per run under profiler/Results/<device>/
RESULTS_DIR=Results/Fold7 ./run_profiles.sh HTP-1Core-ViT-Qwen35-2B-Clean-111226=config_htp_1core_vit_qwen35_2b.json \
                  GPU-Qwen35-2B-Clean-111226=config_gpu_qwen35_2b.json \
                  CPU-Qwen35-2B-Clean-111226=config_cpu_qwen35_2b.json

# Op shares only (profiling build); never use these timings as benchmarks
HTP_PROFILE=1 ./run_profiles.sh HTP-1Core-ViT-Qwen2VL-Prof-111226=config_htp_1core_vit_qwen2vl.json
```

- Before each run the script waits until the phone has cooled (GPU and NPU below 44 °C, CPU below 47 °C, up to 15 minutes).
- Each run goes to `Results/<folder>/` with its config, `run_settings.txt`, `console.txt` and the output files above. Set `RESULTS_DIR=<dir>` to use another folder.
- A folder that already exists is skipped, so an interrupted round can be restarted with the same command.
- Each run is stopped after 1 hour.
- Folder names: `<backend>-<model>-Clean|Prof-MMDDYY`.

### Manual Deploy

```bash
# Push executable
adb push llama.cpp/build-android/bin/mtmd-profile-test /data/local/tmp/

# Push config
adb push configs/config_htp_1core_vit_qwen2vl.json /data/local/tmp/profile-settings.json

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

### Speed Summary for a Round

```bash
python summarize_runs.py Results/Fold7/*-Clean-111226 --out Results/Fold7/summary_111226.md
```

One row per run folder, following the speed protocol (plan.md 0.5; MLPerf Client and on-device VLM studies report time to first token and decode speed separately):

- **Cold** is image 1 (no warmup yet). **Warm** columns are the median [min–max] over images 2..N.
- **TTFT** (time to first token) = image encode + image-token decode + prefill rest (text prompt and overhead). Each part is reported.
- **Decode tok/s** excludes the first token.
- **Image tokens** is the real count from `console.txt`; it differs per model because each uses its own preprocessing.
- **Decode drift** compares the last two warm images with the first two; a negative value flags throttling.
- **Answer tokens** and temperatures are context. Total time is not compared, because it depends on answer length.

Older runs without image markers get encode and image-token columns only when each image is a single slice.

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
| GPU       | `GPUOpenCL`           | N/A          |
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

Result folders are in `Results/<device>/`: `Results/Fold7/` (Samsung Galaxy Z Fold 7 Ultra, Qualcomm Snapdragon 8 Elite) and `Results/Flip6/` (Samsung Galaxy Z Flip 6, Qualcomm Snapdragon 8 Gen 3), set with `RESULTS_DIR` in `run_profiles.sh`.

V2.1 round (2026-11-12): 10 test images, Q4_K_M models with f16 mmproj, clean runs (`-HtpProfile 0`), llama.cpp b12181, Hexagon SDK 6.7.0.1, thinking off, short-answer prompt, sampling 0.6 / 20 / 0.95 / min_p 0.05. Values are medians over images 2–10 with (min–max); image 1 is the cold run and is not counted. Each device's `summary_111226.md` also has cold times, the TTFT split (encode / prefill), decode drift, temperatures and CPU clocks.

**Samsung Galaxy Z Fold 7 Ultra (Qualcomm Snapdragon 8 Elite)**, context size 16384 (`n_ctx`), summary in `Results/Fold7/summary_111226.md`:

| Result Folder                           | Model          | Backend | TTFT, s            | Decode, tok/s     |
| --------------------------------------- | -------------- | ------- | ------------------ | ----------------- |
| `HTP-1Core-ViT-Qwen35-08B-Clean-111226` | Qwen3.5-0.8B   | NPU     | 0.30 (0.20–0.31)   | 54.0 (52.7–64.1)  |
| `HTP-1Core-ViT-Qwen38-Clean-111226`     | Qwen3.8-2B     | NPU     | 0.53 (0.36–0.54)   | 31.0 (30.7–33.0)  |
| `HTP-1Core-ViT-Qwen35-2B-Clean-111226`  | Qwen3.5-2B     | NPU     | 0.53 (0.36–0.55)   | 31.3 (30.9–32.3)  |
| `HTP-1Core-ViT-Qwen3VL-Clean-111226`    | Qwen3-VL-2B    | NPU     | 0.57 (0.38–0.62)   | 33.2 (32.4–35.1)  |
| `HTP-1Core-ViT-Qwen2VL-Clean-111226`    | Qwen2-VL-2B    | NPU     | 0.78 (0.50–0.79)   | 41.3 (40.8–45.3)  |
| `HTP-1Core-ViT-Qwen35-4B-Clean-111226`  | Qwen3.5-4B     | NPU     | 1.00 (0.61–1.28)   | 14.9 (13.9–16.1)  |
| `HTP-1Core-ViT-LFM25-Clean-111226`      | LFM2.5-VL-1.6B | NPU     | 1.46 (0.85–1.60)   | 59.2 (56.8–62.4)  |
| `HTP-1Core-ViT-SmolVLM-Clean-111226`    | SmolVLM2-2.2B  | NPU     | 4.21 (2.21–4.81)   | 33.8 (32.7–38.4)  |
| `HTP-1Core-ViT-Granite-Clean-111226`    | Granite-3.2-2B | NPU     | 13.7 (5.20–16.8)   | 15.1 (14.3–20.0)  |
| `GPU-Qwen35-08B-Clean-111226`           | Qwen3.5-0.8B   | GPU     | 1.33 (0.54–1.56)   | 41.0 (37.8–55.3)  |
| `GPU-Qwen38-Clean-111226`               | Qwen3.8-2B     | GPU     | 4.68 (1.90–4.77)   | 17.4 (17.2–23.5)  |
| `GPU-Qwen35-2B-Clean-111226`            | Qwen3.5-2B     | GPU     | 4.97 (1.68–5.08)   | 17.4 (17.2–27.2)  |
| `GPU-Qwen3VL-Clean-111226`              | Qwen3-VL-2B    | GPU     | 4.82 (1.78–7.75) ⚠ | 14.8 (9.4–21.6) ⚠ |
| `GPU-Qwen2VL-Clean-111226`              | Qwen2-VL-2B    | GPU     | 8.36 (2.61–8.41)   | 16.7 (16.4–24.3)  |
| `GPU-Qwen35-4B-Clean-111226`            | Qwen3.5-4B     | GPU     | 9.27 (3.80–11.3)   | 7.7 (6.7–10.4)    |
| `GPU-LFM25-Clean-111226`                | LFM2.5-VL-1.6B | GPU     | 12.7 (2.63–12.9)   | 29.5 (29.3–49.6)  |
| `GPU-SmolVLM-Clean-111226`              | SmolVLM2-2.2B  | GPU     | 71.8 (36.7–72.6) ❌ | 6.7 (6.5–9.3) ❌   |
| `GPU-Granite-Clean-111226`              | Granite-3.2-2B | GPU     | 84.7 (47.8–100)    | 6.8 (5.7–7.2)     |
| `CPU-Qwen35-08B-Clean-111226`           | Qwen3.5-0.8B   | CPU     | 6.97 (1.77–8.09)   | 15.9 (13.7–23.5)  |
| `CPU-Qwen38-Clean-111226`               | Qwen3.8-2B     | CPU     | 21.1 (11.3–21.5)   | 9.0 (8.9–9.3)     |
| `CPU-Qwen35-2B-Clean-111226`            | Qwen3.5-2B     | CPU     | 21.1 (8.53–23.0)   | 9.1 (8.7–9.8)     |
| `CPU-Qwen3VL-Clean-111226`              | Qwen3-VL-2B    | CPU     | 20.9 (8.61–22.6)   | 10.8 (10.4–13.4)  |
| `CPU-Qwen2VL-Clean-111226`              | Qwen2-VL-2B    | CPU     | 42.5 (19.8–46.5)   | 11.1 (10.3–12.8)  |
| `CPU-Qwen35-4B-Clean-111226`            | Qwen3.5-4B     | CPU     | 33.2 (19.4–34.4)   | 4.7 (4.6–4.9)     |
| `CPU-LFM25-Clean-111226`                | LFM2.5-VL-1.6B | CPU     | 22.2 (7.25–23.7)   | 19.1 (17.8–24.7)  |
| `CPU-SmolVLM-Clean-111226`              | SmolVLM2-2.2B  | CPU     | 47.0 (30.0–47.6)   | 13.0 (12.6–13.6)  |
| `CPU-Granite-Clean-111226`              | Granite-3.2-2B | CPU     | 271 (135–284)      | 4.5 (4.1–6.0)     |

⚠ On GPU, Qwen3-VL sometimes does not stop and gives very long answers (3 of the 9 warm images); the README uses the average of the other 6 images (4.1 s, 16.2 tok/s). ❌ On GPU, SmolVLM2 gives garbage output, so its GPU numbers are not used; it works on NPU and CPU.

**Samsung Galaxy Z Flip 6 (Qualcomm Snapdragon 8 Gen 3)**, context size 4096 (`n_ctx`), summary in `Results/Flip6/summary_111226.md`:

| Result Folder                           | Model          | Backend | TTFT, s          | Decode, tok/s    |
| --------------------------------------- | -------------- | ------- | ---------------- | ---------------- |
| `HTP-1Core-ViT-Qwen35-08B-Clean-111226` | Qwen3.5-0.8B   | NPU     | 0.48 (0.33–0.50) | 35.5 (34.7–41.1) |
| `HTP-1Core-ViT-Qwen38-Clean-111226`     | Qwen3.8-2B     | NPU     | 0.92 (0.61–0.99) | 19.3 (18.3–20.1) |
| `HTP-1Core-ViT-Qwen35-2B-Clean-111226`  | Qwen3.5-2B     | NPU     | 0.91 (0.61–0.93) | 19.9 (19.3–20.4) |
| `HTP-1Core-ViT-Qwen3VL-Clean-111226`    | Qwen3-VL-2B    | NPU     | 0.91 (0.58–0.94) | 20.3 (19.7–23.3) |
| `HTP-1Core-ViT-Qwen2VL-Clean-111226`    | Qwen2-VL-2B    | NPU     | 1.21 (0.81–1.30) | 26.5 (23.9–28.4) |
| `HTP-1Core-ViT-LFM25-Clean-111226`      | LFM2.5-VL-1.6B | NPU     | 3.50 (1.85–3.52) | 34.6 (34.0–37.8) |
| `HTP-1Core-ViT-SmolVLM-Clean-111226`    | SmolVLM2-2.2B  | NPU     | 7.90 (4.76–7.95) | 19.2 (18.8–21.3) |
| `HTP-1Core-ViT-Granite-Clean-111226`    | Granite-3.2-2B | NPU     | 26.0 (10.9–35.1) | 8.9 (8.5–11.7)   |
| `CPU-Qwen35-08B-Clean-111226`           | Qwen3.5-0.8B   | CPU     | 4.85 (2.24–11.9) | 16.7 (10.2–22.1) |
| `CPU-Qwen38-Clean-111226`               | Qwen3.8-2B     | CPU     | 30.8 (7.11–33.8) | 6.0 (5.9–10.7)   |
| `CPU-Qwen35-2B-Clean-111226`            | Qwen3.5-2B     | CPU     | 19.5 (7.50–31.7) | 9.9 (5.6–9.9)    |
| `CPU-Qwen3VL-Clean-111226`              | Qwen3-VL-2B    | CPU     | 31.3 (5.93–33.9) | 6.8 (6.1–16.8)   |
| `CPU-Qwen2VL-Clean-111226`              | Qwen2-VL-2B    | CPU     | 61.0 (14.5–65.9) | 7.6 (7.1–12.8)   |
| `CPU-Qwen35-4B-Clean-111226`            | Qwen3.5-4B     | CPU     | 31.7 (19.7–31.8) | 4.6 (4.6–4.7)    |
| `CPU-LFM25-Clean-111226`                | LFM2.5-VL-1.6B | CPU     | 32.6 (5.83–35.5) | 12.3 (10.4–29.6) |
| `CPU-SmolVLM-Clean-111226`              | SmolVLM2-2.2B  | CPU     | 40.6 (23.0–41.0) | 13.1 (12.9–13.2) |
| `CPU-Granite-Clean-111226`              | Granite-3.2-2B | CPU     | 245 (133–245)    | 6.0 (5.9–7.2)    |

The Flip 6 NPU needs a context size of 4096 (16384 gave IOVA mapping failures); the earlier 16384 runs are in `Results/Flip6/ctx16384/`. Qwen3.5-4B does not fit on the Flip 6 NPU. Q4_K_M models are not supported on the Flip 6 GPU (Adreno 750). The Flip 6 throttles more than the Fold 7, so CPU numbers vary a lot over a run.

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
