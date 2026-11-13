# LLAMA VLM Android

A mobile live camera & screen capture analysis tool that uses Vision Language Models (VLM) to analyze what's on your screen. Remote mode uses [llama.cpp](https://github.com/ggml-org/llama.cpp) server API; local mode uses a custom JNI wrapper built on top of llama.cpp's [mtmd-cli](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd) multimodal inference engine. Offline text-to-speech powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

Supports local acceleration with CPU (ARM cores), GPU (Adreno via [OpenCL](https://www.khronos.org/opencl/)), or NPU (Hexagon HTP, using [Hexagon SDK](https://softwarecenter.qualcomm.com/)).

**Development:** This repository was fully developed using [Claude CLI](https://claude.ai/code) with local llama.cpp server running the `unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL` model ([HuggingFace](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF)).

![Platform](https://img.shields.io/badge/Platform-Android-green.svg)
![API](https://img.shields.io/badge/API-31%2B-brightgreen.svg)
![Min SDK](https://img.shields.io/badge/Min%20SDK-31-blue.svg)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Backend Modes](#backend-modes)
  - [Capture Sources](#capture-sources)
  - [Remote Mode](#remote-mode)
  - [Local Mode](#local-mode)
- [Configuration](#configuration)
- [Model Setup](#model-setup)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Build Documentation](#build-documentation)

---

## Features

### Core Capabilities

- **Screen Capture**: Real-time screen capture with overlay controls
- **Camera Capture**: Front/back camera streaming with tap-to-capture
- **Visual Analysis**: Ask questions about what's on your screen
- **Offline TTS**: Text-to-speech output using offline engine
- **Streaming Response**: Real-time token streaming from VLM
- **Thinking Models**: `<think>` blocks are hidden from the display and from speech

### Backend Options

| Mode   | Description                         | Use Case                                |
|--------|-------------------------------------|-----------------------------------------|
| REMOTE | Connects to remote llama.cpp server | Fast setup, cloud processing            |
| LOCAL  | On-device inference with llama.cpp  | Offline use, privacy, no network needed |


### Local Backend Acceleration

| Backend | Hardware        | Decode Range, 9 Models |
| ------- | --------------- | ---------------------- |
| CPU     | ARM cores       | 4.5–19 tk/s            |
| NPU     | Hexagon HTP     | 15–59 tk/s             |
| GPU     | Adreno (OpenCL) | 7–41 tk/s              |

See [Local Inference Benchmarks](#local-inference-benchmarks) for each model. These were running on the Samsung Galaxy Z Fold 7 Ultra (Qualcomm Snapdragon 8 Elite).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LLAMA VLM Android                       │
├─────────────────────────────────────────────────────────────┤
│  MainActivity                                               │
│  ├── Mode Selection (REMOTE / LOCAL)                        │
│  ├── Capture Source (Screen / Camera)                       │
│  └── Settings Configuration                                 │
├─────────────────────────────────────────────────────────────┤
│  ScreenCaptureService   │  CameraCaptureActivity            │
│  ├── Screen Capture     │  ├── Camera Preview               │
│  ├── Overlay Controls   │  ├── Frame Capture                │
│  └── Inference Routing  │  └── YUV→RGB Conversion           │
├─────────────────────────────────────────────────────────────┤
│  VlmAnalyzer                                                │
│  ├── Image Pre-Processing (LOCAL/REMOTE)                    │
│  ├── API Client (remote HTTP)                               │
│  └── Local Inference Backend (LOCAL mode)                   │
├─────────────────────────────────────────────────────────────┤
│  Local Inference Backend (LOCAL mode)                       │
│  ├── libmtmd-inference-jni.so (JNI wrapper)                 │
│  ├── libllama.so + libmtmd.so (multimodal inference)        │
│  └── Backends: CPU / GPU (OpenCL) / NPU (Hexagon HTP)       │
├─────────────────────────────────────────────────────────────┤
│  TtsHelper (Offline TTS)                                    │
│  └── Text-to-speech for results                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Quick Start

1. **Download APK**: Get the latest release from the releases page
2. **Install**: Enable "Install from Unknown Sources" if needed
3. **Launch**: Open the app and grant required permissions

### Build from Source

See [SETUP.md](SETUP.md) for complete build instructions.

**Prerequisites:**
- MSYS2 UCRT64 environment (required for building llama.cpp)
- Android SDK with NDK 27.3.13750724
- Java JDK 17+
- Hexagon SDK 6.7.0.1 - for NPU support. Qualcomm no longer offers a direct download, so install it with their download tool from the [Qualcomm Software Center](https://softwarecenter.qualcomm.com/)

**Quick Build:**
```bash
# MSYS2 UCRT64 terminal
cd VLM-Analyze-Android
source ./setup-config.sh    # Configure paths
./buildAPK.sh -All -install # Build and install
```

---

## Usage

![Main Activity](docs/images/Main-Activity-Camera-Local.jpg)
*Main activity - select backend mode (REMOTE/LOCAL) and capture source (Screen/Camera)*

### Backend Modes

The app supports two backend modes for VLM inference:

#### REMOTE Mode (Cloud API)

Connects to a remote llama.cpp server running on another machine.

**Setup:**
1. Launch a llama.cpp server with multimodal support:
   
   **Example with Qwen3.5-122B (recommended for powerful servers):**
   ```bash
   llama-server -hf unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL \
                -c 262144 --host 0.0.0.0 -ngl all --no-mmap \
                --fit-ctx 32768 --flash-attn on \
                --reasoning-format auto --temp 0.6 --top-p 0.95 \
                --top-k 20 --min-p 0.00 \
                --presence_penalty 0.0 --repeat-penalty 1.0 \
                --chat-template-file qwen3.5_chat_template.jinja
   ```
   
   **Example with Qwen2-VL-2B (lightweight):**
   ```bash
   llama-server -hf bartowski/Qwen2-VL-2B-Instruct-GGUF \
                -m Qwen2-VL-2B-Bartowski-Q4_K_M.gguf \
                --mmproj mmproj-f16.gguf \
                -c 16384 --host 0.0.0.0 -ngl 999
   ```
   
   **Supported models:** Any gguf multimodal model compatible with llama.cpp

2. In the app:
   - Tap "REMOTE" mode button
   - Tap "Settings" to configure API URL
   - Default: `http://localhost:8080/v1/chat/completions`

**Pros:**
- Broad Compatibility
- Can use larger models (70B+, 122B+)
- Best quality with powerful models

**Cons:**
- Requires network connection
- Server setup required

**Configuration:**
- **API URL**: Endpoint for llama.cpp server
- **Model**: Model name (for display/reference)

**Note:** Image Scale is controlled via slider. Separate values for Screen and Camera capture. Lower values = faster processing, less detail.

**Usage:**
1. Select "REMOTE" mode
2. Configure API URL in Settings
3. Start capture (screen or camera)
4. Tap capture to send image to server
5. View streaming response

![Remote Settings](docs/images/Main-Activity-Remote-Settings.jpg)
*Remote mode settings - configure API URL and model name*

#### LOCAL Mode (On-Device)

Runs inference directly on your Android device using llama.cpp.

**Setup:**
1. See [Model Setup](#model-setup) for pushing models to device
2. In the app:
   - Tap "LOCAL" mode button
   - Tap "Local Settings" to configure
   - Select model and mmproj paths
   - Choose backend (CPU/GPU/NPU)
   - Tap "Start Local" to load the model

**Pros:**
- Works offline
- No network latency
- Privacy (data stays on device)

**Cons:**
- Limited by device capacity
- Quality may drop

**Configuration:**
- **Model Path**: Path to `.gguf` model file
- **MMProj Path**: Path to multimodal projector file
- **Backend**: CPU, GPU (Adreno), or NPU (Hexagon HTP)
- **Context Size**: Token context window (default: 16384)
- **Batch Size**: Processing batch size (default: 512)
- **HTP Sessions**: Number of Hexagon sessions (1-4, for NPU). 1 is recommended
- **Inference Params**: top-k, top-p, temperature, repetition penalty

**NPU mode:** The language model and the vision encoder both run on Hexagon HTP. The encoder always uses `HTP0`, so changing HTP Sessions only affects the language model. In CPU and GPU modes the encoder stays on that backend.

**Usage:**
1. Select "LOCAL" mode
2. Tap "Local Settings"
3. Configure model, mmproj, and backend
4. Tap "Start Local" to load model
5. Start capture (Screen or Camera) and analyze

![Local Settings](docs/images/Main-Activity-Local-Local-Settings.jpg)
*Local mode settings - configure model path, mmproj, backend (CPU/GPU/NPU), and inference parameters*

**Note:** Both Screen and Camera capture sources work in REMOTE and LOCAL modes.

### Capture Sources

#### Screen Capture

Analyze what's displayed on your screen. Works in both REMOTE and LOCAL modes.

**Steps:**
1. Select "Screen" in capture source
2. Tap "Start Analyzer"
3. Grant screen capture permission (one-time)
4. Grant overlay permission (for floating controls)
5. Tap "Capture" on the floating overlay to analyze

**Features:**
- Real-time screen capture
- Floating overlay with capture button
- Results displayed on overlay
- Voice output via TTS

![Screen Capture Overlay](docs/images/Screen-Capture-Overlay.jpg)
*Floating overlay during screen capture - tap camera icon to capture, or microphone to ask voice questions like "What is on my screen?"*

#### Camera Capture

Analyze what the camera sees. Works in both REMOTE and LOCAL modes.

**Steps:**
1. Select "Camera" in capture source
2. Tap "Start Analyzer"
3. Grant camera permission
4. Camera preview opens
5. Tap capture button to analyze current frame

**Features:**
- Live camera preview at 30 FPS
- Front/back camera toggle
- Tap-to-capture analysis
- Streaming pauses during analysis
- Auto-detected camera resolution

![Camera Preview](docs/images/Camera-Preview.jpg)
*Same functionality as screen capture (camera icon to capture, microphone for voice questions) plus camera toggle icon to swap front/back camera*

![Analysis Result](docs/images/Result-Camera-Overlay.jpg)
*Analysis result displayed on overlay after capture*

![Result in Main Activity](docs/images/Result-Main-Activity.jpg)
*Full analysis text returned by the VLM model, displayed in the main activity*

---

## Configuration

### Main Settings (REMOTE Mode)

| Setting | Description               | Default                                    |
|---------|---------------------------|--------------------------------------------|
| API URL | Remote server endpoint    | `http://localhost:8080/v1/chat/completions`|
| Model   | Model name (display only) | `unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL`|

### General Settings (Both Modes)

| Setting              | Description                        | Default |
| -------------------- | ---------------------------------- | ------- |
| Image Scale (Screen) | Screen capture resolution (slider) | 0.4     |
| Image Scale (Camera) | Camera capture resolution (slider) | 0.25    |


### Local Settings (LOCAL Mode)

| Setting            | Description            | Default                                                                                                                                            |
| ------------------ | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model Path         | Path to `.gguf` model  | -                                                                                                                                                  |
| MMProj Path        | Path to projector file | -                                                                                                                                                  |
| Backend            | CPU / GPU / NPU        | CPU                                                                                                                                                |
| Context Size       | Token context window   | `16384`                                                                                                                                            |
| Batch Size         | Processing batch size  | `512`                                                                                                                                              |
| HTP Sessions       | Hexagon sessions (NPU) | `1`  * HTP Sessions is Device-dependent: Samsung Galaxy Z Fold 7 Ultra supports up to 4, 1 ran faster than 4. Samsung Galaxy Z Flip 6 supports 1.* |
| Image Min Tokens   | Minimum image tokens   | `-1` (auto)                                                                                                                                        |
| Top-k              | Sampling top-k         | `40`                                                                                                                                               |
| Top-p              | Sampling top-p         | `0.95`                                                                                                                                             |
| Temperature        | Sampling temperature   | `0.80`                                                                                                                                             |
| Repetition Penalty | Repetition penalty     | `1.00`                                                                                                                                             |

 

### Other Settings

| Setting          | Description                | Default |
|------------------|----------------------------|---------|
| Save Audio       | Save audio recordings      | `false` |
| Save Screenshots | Save captured screenshots  | `false` |

### Logging

Enable debug or info logs for troubleshooting:
- **Debug Logs**: Detailed operation logs
- **Info Logs**: Component-specific logs

### Speed Display

- **LOCAL mode**: The t/s value is decode speed, the same value as `Decode throughput` in logcat. It does not include prefill.
- **REMOTE mode**: The API gives no decode timing, so t/s is (prompt + completion tokens) / total time. Do not compare it directly with LOCAL mode.
- **Thinking models**: Thinking tokens count towards completion tokens and t/s, even though the text is hidden.

---

## Model Setup

### Required Files

For multimodal VLM inference, you need:

1. **Model File** (`.gguf`): The VLM model weights
2. **MMProj File** (`.gguf`): Multimodal projector for image encoding

### Supported Models

All llama.cpp models should be supported. These 9 models are tested on NPU, GPU and CPU. Other llama.cpp multimodal (mtmd) models were tested on Samsung Galaxy Z Fold 7 Ultra (Qualcomm Snapdragon 8 Elite).

| Model                      | GGUF Source (Hugging Face)                                                          | Model Q4_K_M | mmproj f16 | Notes                                            |
| -------------------------- | ----------------------------------------------------------------------------------- | ------------ | ---------- | ------------------------------------------------ |
| Qwen2-VL-2B-Instruct       | [bartowski](https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF)             | 0.99 GB      | 1.33 GB    |                                                  |
| Qwen3-VL-2B-Instruct       | [unsloth](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF)                 | 1.11 GB      | 0.82 GB    | GPU: sometimes very long answers                 |
| Qwen3.5-0.8B               | [unsloth](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)                         | 0.53 GB      | 0.21 GB    |                                                  |
| Qwen3.5-2B                 | [unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)                           | 1.28 GB      | 0.67 GB    |                                                  |
| Qwen3.5-4B                 | [unsloth](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)                           | 2.74 GB      | 0.67 GB    |                                                  |
| Qwen3.8-2B-Distill-heretic | [mradermacher](https://huggingface.co/mradermacher/Qwen3.8-2B-Distill-heretic-GGUF) | 1.27 GB      | 0.67 GB    | Thinking model (tested with thinking off)        |
| LFM2.5-VL-1.6B             | [lmstudio-community](https://huggingface.co/lmstudio-community/LFM2.5-VL-1.6B-GGUF) | 0.73 GB      | 0.85 GB    |                                                  |
| SmolVLM2-2.2B-Instruct     | [ggml-org](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF)             | 1.11 GB      | 0.87 GB    | GPU: garbage output                              |
| Granite-Vision-3.2-2B      | [ibm-research](https://huggingface.co/ibm-research/granite-vision-3.2-2b-GGUF)      | 1.55 GB      | 0.89 GB    | About 3,600 image tokens per image: slow prefill |


### Pushing Models to Device

```bash
# Create model directory
adb shell "mkdir -p /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/"

# Push model files
adb push Qwen2-VL-2B-Bartowski-Q4_K_M.gguf /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/
adb push mmproj-f16.gguf /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/

# Verify
adb shell "ls -lh /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/"
```

---

## Performance

### Local Inference Benchmarks

**Test Configuration:**
- Device: Samsung Galaxy Z Fold 7 Ultra (Adreno 830, Hexagon v79)
- Build: llama.cpp b12181, Hexagon SDK 6.7.0.1
- Models: the 9 [tested models](#supported-models), Q4_K_M with f16 mmproj
- Context size 16384 (`n_ctx`)
- 10 test images, image tokens at each model's default
- Thinking off, Jinja template on, short-answer prompt
- Sampling: temperature 0.6, top_k 20, top_p 0.95, min_p 0.05
- NPU uses 1 HTP session with the vision encoder on HTP
- The phone is not rooted, so the CPU clock can't be fixed. After image 1 the thermal limit holds the CPU at about 1 GHz
- Profiler with HTP profiling off (`-HtpProfile 0`), after a cooldown. See [PROFILE_TEST_README.md](profiler/PROFILE_TEST_README.md)

Values are medians over images 2–10 (warm). TTFT (time to first token) covers image encode and prefill.

| Model          | Image Tokens | TTFT NPU | TTFT GPU | TTFT CPU | Decode NPU | Decode GPU  | Decode CPU |
| -------------- | ------------ | -------- | -------- | -------- | ---------- | ----------- | ---------- |
| Qwen3.5-0.8B   | 252          | 0.30 s   | 1.3 s    | 7.0 s    | 54.0 tk/s  | 41.0 tk/s   | 15.9 tk/s  |
| Qwen3.8-2B     | 252          | 0.53 s   | 4.7 s    | 21.1 s   | 31.0 tk/s  | 17.4 tk/s   | 9.0 tk/s   |
| Qwen3.5-2B     | 252          | 0.53 s   | 5.0 s    | 21.1 s   | 31.3 tk/s  | 17.4 tk/s   | 9.1 tk/s   |
| Qwen3-VL-2B    | 252          | 0.57 s   | 4.1 s ⚠  | 20.9 s   | 33.2 tk/s  | 16.2 tk/s ⚠ | 10.8 tk/s  |
| Qwen2-VL-2B    | 315          | 0.78 s   | 8.4 s    | 42.5 s   | 41.3 tk/s  | 16.7 tk/s   | 11.1 tk/s  |
| Qwen3.5-4B     | 252          | 1.0 s    | 9.3 s    | 33.2 s   | 14.9 tk/s  | 7.7 tk/s    | 4.7 tk/s   |
| LFM2.5-VL-1.6B | 252          | 1.5 s    | 12.7 s   | 22.2 s   | 59.2 tk/s  | 29.5 tk/s   | 19.1 tk/s  |
| SmolVLM2-2.2B  | 243          | 4.2 s    | ❌       | 47.0 s   | 33.8 tk/s  | ❌          | 13.0 tk/s  |
| Granite-3.2-2B | 3645         | 13.7 s   | 84.7 s   | 271 s    | 15.1 tk/s  | 6.8 tk/s    | 4.5 tk/s   |

*⚠ On GPU, Qwen3-VL sometimes does not stop and gives very long answers (3 of the 9 warm images); its GPU numbers are the average of the other 6 images. It works on NPU and CPU. ❌ On GPU, SmolVLM2 gives garbage output, so its GPU measurement is dropped; it works on NPU and CPU. GPU decode slows by 20–37% over a run from thermal throttling. The first image after loading a model is slower, so it is not counted.*
*⚠ On GPU, Qwen3-VL sometimes does not stop and gives very long answers (3 of the 9 warm images); its GPU numbers are the average of the other 6 images. It works on NPU and CPU. ❌ On GPU, SmolVLM2 gives garbage output, so its GPU measurement is dropped; it works on NPU and CPU. GPU decode slows by 20–37% over a run from thermal throttling. The first image after loading a model is slower, so it is not counted.*

**Samsung Galaxy Z Flip 6 (Qualcomm Snapdragon 8 Gen 3; Adreno 750, Hexagon v75)**, same build and settings as above, with a context size of 4096 (`n_ctx`):

| Model          | Image Tokens | TTFT NPU | TTFT CPU | Decode NPU | Decode CPU |
| -------------- | ------------ | -------- | -------- | ---------- | ---------- |
| Qwen3.5-0.8B   | 252          | 0.48 s   | 4.8 s    | 35.5 tk/s  | 16.7 tk/s  |
| Qwen3.8-2B     | 252          | 0.92 s   | 30.8 s   | 19.3 tk/s  | 6.0 tk/s   |
| Qwen3.5-2B     | 252          | 0.91 s   | 19.5 s   | 19.9 tk/s  | 9.9 tk/s   |
| Qwen3-VL-2B    | 252          | 0.91 s   | 31.3 s   | 20.3 tk/s  | 6.8 tk/s   |
| Qwen2-VL-2B    | 315          | 1.2 s    | 61.0 s   | 26.5 tk/s  | 7.6 tk/s   |
| Qwen3.5-4B     | 252          | ❌       | 31.7 s   | ❌         | 4.6 tk/s   |
| LFM2.5-VL-1.6B | 252          | 3.5 s    | 32.6 s   | 34.6 tk/s  | 12.3 tk/s  |
| SmolVLM2-2.2B  | 243          | 7.9 s    | 40.6 s   | 19.2 tk/s  | 13.1 tk/s  |
| Granite-3.2-2B | 3645         | 26.0 s   | 245 s    | 8.9 tk/s   | 6.0 tk/s   |

*❌ Qwen3.5-4B does not fit on the Flip 6 NPU. GPU: Q4_K_M models are not supported on the Adreno 750. The Flip 6 throttles more than the Fold 7, so CPU numbers vary a lot over a run.*


### Factors Affecting Performance

1. **Image Scale**: Lower scale = faster processing
2. **Model Size**: Smaller quantizations = faster
3. **Encoder Placement**: Running the vision encoder on HTP cuts prefill to under 1 second and avoids the GPU thermal cap
4. **Context Size**: Larger context = more memory, slower

### Tips for Better Performance

1. Enable GPU/NPU if available, and use 1 HTP session in NPU mode
2. Reduce image scale for faster capture
3. Set appropriate context size for your use case

---

## Troubleshooting

### "Overlay permission required"

- Go to Settings → Apps → Special Access → Display over other apps
- Enable for VLM-Analyze app

### "Screen capture permission denied"

- Restart the app
- Grant permission when prompted
- Permission persists across app restarts

### "Model not found"

- Verify models are pushed to `/data/local/tmp/llama.cpp/models/`
- Check file permissions: `adb shell "ls -lh /data/local/tmp/llama.cpp/models/"`
- Use full paths in Local Settings

### "Local inference failed to load"

- Check available memory (close other apps)
- Verify model and mmproj are compatible
- Try CPU backend if GPU/NPU fails
- Check logcat for detailed errors: `adb logcat -s VLM-Inference:V`

### "Slow performance"

- Reduce image scale factor
- Use smaller model quantization
- Enable GPU or NPU backend
- Reduce context size if not needed

### Monitoring Logs

```bash
# All VLM logs
adb logcat -s VLM-Application:V VLM-MainActivity:V VLM-Service:V VLM-Capture:V VLM-Inference:V mtmd-inference:V *:F

# Camera logs
adb logcat -s VLM-Camera:V VLM-CameraPreview:V VLM-CameraCapture:V *:F

# Enable debug logs in app settings first

# Vision encoder device and per-image timing
adb logcat | grep -E "Vision encoder device|CLIP using|Prefill time|Decode throughput"
```

In NPU mode, a correct setup logs `Vision encoder device: HTP0` and `CLIP using HTP0 backend` when the model loads.

---

## Build Documentation

For detailed build instructions:

- **[SETUP.md](SETUP.md)** - Complete setup and build guide
- **[BUILD_LLAMACPP_ANDROID_README.md](BUILD_LLAMACPP_ANDROID_README.md)** - llama.cpp build details (in case needed)
- **[PROFILE_TEST_README.md](profiler/PROFILE_TEST_README.md)** - On-device profiling and timing runs

---

## License

MIT License - See LICENSE file for details

---

## Credits

- **llama.cpp** - https://github.com/ggml-org/llama.cpp
- **Qwen2-VL-2B-Instruct GGUF** - https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF (quantized by Bartowski)
- **Qwen3-VL-2B-Instruct GGUF** - https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF (quantized by Unsloth)
- **Qwen3.5-0.8B GGUF** - https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF (quantized by Unsloth)
- **Qwen3.5-2B GGUF** - https://huggingface.co/unsloth/Qwen3.5-2B-GGUF (quantized by Unsloth)
- **Qwen3.5-4B GGUF** - https://huggingface.co/unsloth/Qwen3.5-4B-GGUF (quantized by Unsloth)
- **Qwen3.8-2B-Distill-heretic GGUF** - https://huggingface.co/mradermacher/Qwen3.8-2B-Distill-heretic-GGUF (quantized by mradermacher; original model: https://huggingface.co/htb-ac-1424625/Qwen3.8-2B-Distill-heretic)
- **LFM2.5-VL-1.6B GGUF** - https://huggingface.co/lmstudio-community/LFM2.5-VL-1.6B-GGUF (quantized by LM Studio Community)
- **SmolVLM2-2.2B-Instruct GGUF** - https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF (quantized by ggml-org)
- **Granite-Vision-3.2-2B GGUF** - https://huggingface.co/ibm-research/granite-vision-3.2-2b-GGUF (IBM Research)
- **Qwen3.5-122B GGUF** - https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF (quantized by Unsloth, remote mode)
- **sherpa-onnx** - https://github.com/k2-fsa/sherpa-onnx (TTS)
- **Claude CLI** - https://claude.ai/code (development assistant)
- **OpenCL** - https://www.khronos.org/opencl/ (GPU acceleration)
- **Hexagon SDK** - https://www.qualcomm.com/ (NPU acceleration)
- **OpenCV** - https://github.com/opencv/opencv (Image Processing)
