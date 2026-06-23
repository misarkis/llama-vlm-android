# mtmd-inference-api

A clean C++ API for multimodal inference using llama.cpp's mtmd (multimodal) functionality. This library wraps the console-based mtmd-cli functionality into reusable API calls.

**Key Features:**
- Images passed as **bitmap data in memory** (RGB format), not file paths
- Prompts passed as **system and user strings in memory**
- Load model once, run multiple inference calls
- Stream token-by-token output via callbacks
- Support for CPU, GPU (OpenCL), and NPU (Hexagon) backends

## Building

### Android

```bash
cd llama.cpp
cmake -B build-android -DLLAMA_BUILD_SERVER=ON
cmake --build build-android --target mtmd-inference-api mtmd-inference-test -j4
```

**Outputs:**
- Library: `build-android/tools/mtmd-inference-api/libmtmd-inference-api.a`
- Test executable: `build-android/bin/mtmd-inference-test`
- Example executable: `build-android/bin/mtmd-inference-example`

### Required Dependencies

- `llama` - llama.cpp core library
- `mtmd` - multimodal library
- `ggml` - tensor library
- `llama-common` - common utilities
- `Threads::Threads` - threading library

## API Reference

### Initialization

```cpp
#include "mtmd-inference-api.h"

// Step 1: Initialize parameters
mtmd_inference_params_t params;
mtmd_inference_params_default(&params);

// Configure model paths
params.model.path = "/data/local/tmp/models/Qwen2-VL-2B.gguf";
params.model.mmproj_path = "/data/local/tmp/models/mmproj-f16.gguf";

// Configure device (choose one):
params.device = "CPU";                    // CPU only
// params.device = "GPU0";                // GPU (OpenCL)
// params.device = "HTP0,HTP1,HTP2,HTP3"; // NPU (Hexagon)

// Configure GPU layers (0 = CPU, 999 = all layers)
params.n_gpu_layers = 999;

// Configure context
params.n_ctx = 16384;
params.n_batch = 512;

// Configure generation
params.n_predict = 128;  // Max tokens to generate

// Configure sampling
params.sampling.temp = 0.8f;
params.sampling.top_k = 40;
params.sampling.top_p = 0.95f;

// Step 2: Initialize the inference engine
mtmd_inference_context_t* ctx = nullptr;
mtmd_api_status status = mtmd_inference_init(params, &ctx, nullptr, nullptr);

if (status != MTMD_API_SUCCESS) {
    // Handle error
}
```

### Running Inference

```cpp
// Step 3: Prepare image data (RGB format in memory)
mtmd_image_data_t image;
image.width = 256;
image.height = 256;
image.data = rgb_data;  // Pointer to RGBRGBRGB... data (width * height * 3 bytes)

// Step 4: Prepare prompts
mtmd_prompt_params_t prompt;
prompt.system_prompt = "You are a helpful assistant.";  // Optional
prompt.user_prompt = "What is in this image? <__media__>";  // Include <__media__> marker
prompt.add_ass_prefix = true;

// Step 5: Run inference
mtmd_inference_result_t result;

// Token callback for streaming (optional)
static void on_token(const char* token, void* user_data) {
    printf("%s", token);
}

status = mtmd_inference_run(
    ctx,
    &image,       // Array of image data
    1,            // Number of images
    prompt,       // Prompt parameters
    &result,
    on_token,     // Token callback (optional)
    nullptr       // User data for callback
);

if (status == MTMD_API_SUCCESS) {
    printf("Response: %s\n", result.text.c_str());
    printf("Tokens: %d\n", result.n_tokens);
    printf("Time: %.0f ms\n", result.eval_time_ms);
    printf("Throughput: %.2f tokens/sec\n", result.tokens_per_second);
}
```

### Cleanup

```cpp
mtmd_inference_free(ctx);
```

## Data Formats

### Image Data (mtmd_image_data_t)

```cpp
typedef struct {
    uint32_t       width;           // Image width
    uint32_t       height;          // Image height
    const uint8_t* data;            // Image data in RGB format
} mtmd_image_data_t;
```

**RGB Format:**
- Data layout: `R G B R G B R G B ...` (width * height * 3 bytes)
- Each channel: 8-bit (0-255)
- No alpha channel
- Row-major order (top to bottom, left to right)

**Example: Loading an image with stb_image**

```cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int width, height, channels;
unsigned char* data = stbi_load("image.png", &width, &height, &channels, 3);

mtmd_image_data_t image;
image.width = width;
image.height = height;
image.data = data;  // Use directly (stbi_load with 3 channels returns RGB)

// ... use image ...

stbi_free(data);  // Free when done
```

### Prompts (mtmd_prompt_params_t)

```cpp
typedef struct {
    std::string system_prompt;  // System prompt (optional, empty = no system prompt)
    std::string user_prompt;    // User prompt (required)
    bool        add_ass_prefix; // Add assistant prefix
} mtmd_prompt_params_t;
```

**Prompt Construction:**
The API constructs the full prompt as:
```
[system_prompt] + [user_prompt with <__media__> marker] + [assistant prefix]
```

Example:
- System: "You are a helpful assistant."
- User: "What is in this image? <__media__>"

Result:
```
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is in this image? <__media__><|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

## Running the Test on Device

### Deploy to Android Device

```bash
# Copy the test executable
adb push build-android/bin/mtmd-inference-test /data/local/tmp/

# Copy required shared libraries
adb push build-android/bin/libllama.so /data/local/tmp/
adb push build-android/bin/libmtmd.so /data/local/tmp/
adb push build-android/bin/libggml*.so /data/local/tmp/
adb push build-android/bin/libllama-common.so /data/local/tmp/

# Copy model files (if not already on device)
adb push models/Qwen2-VL-2B-Q4_K_M.gguf /data/local/tmp/models/
adb push models/mmproj-f16.gguf /data/local/tmp/models/

# Copy test image
adb push test.png /data/local/tmp/
```

### Run on Device

```bash
adb shell
cd /data/local/tmp
export LD_LIBRARY_PATH=.

# Test with image file
./mtmd-inference-test \
    --model /data/local/tmp/models/Qwen2-VL-2B-Q4_K_M.gguf \
    --mmproj /data/local/tmp/models/mmproj-f16.gguf \
    --image /data/local/tmp/test.png \
    --system "You are a helpful assistant." \
    --user "What is in this image?" \
    --device "HTP0,HTP1,HTP2,HTP3" \
    --n-gpu-layers 999 \
    --n-predict 128

# Test with generated bitmap (no image file needed)
./mtmd-inference-test \
    --model /data/local/tmp/models/Qwen2-VL-2B-Q4_K_M.gguf \
    --mmproj /data/local/tmp/models/mmproj-f16.gguf \
    --bitmap-test \
    --user "Describe this gradient pattern" \
    --device "HTP0,HTP1,HTP2,HTP3"
```

### Test Command Options

```
Usage: ./mtmd-inference-test [options]

Options:
  --model <path>        Path to model GGUF file (required)
  --mmproj <path>       Path to mmproj GGUF file (required)
  --image <path>        Path to test image (optional, use with --bitmap-test)
  --bitmap-test         Use generated test bitmap instead of file
  --system <text>       System prompt (optional)
  --user <text>         User prompt (required if not --bitmap-test)
  --device <dev>        Device: CPU, GPU0, HTP0,HTP1,HTP2,HTP3 (default: CPU)
  --n-gpu-layers <n>    GPU layers (0=CPU, 999=all, default: 999)
  --n-ctx <n>           Context size (default: 16384)
  --n-predict <n>       Max tokens to generate (default: 128)
  --temp <f>            Temperature (default: 0.8)
  --help                Show help
```

## Parameter Structures

### mtmd_inference_params_t

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model.path` | string | "" | Path to model GGUF file |
| `model.mmproj_path` | string | "" | Path to multimodal projector |
| `device` | string | "CPU" | Device string (CPU, GPU0, HTP0, etc.) |
| `n_gpu_layers` | int32 | 999 | Layers to offload (0=CPU, 999=all) |
| `n_ctx` | int32 | 16384 | Context size |
| `n_batch` | int32 | 512 | Batch size for prompt processing |
| `n_predict` | int32 | -1 | Max tokens to generate (-1=unlimited) |
| `cpuparams.n_threads` | int32 | 4 | Number of threads |
| `sampling.temp` | float | 0.8 | Temperature |
| `sampling.top_k` | int32 | 40 | Top-k sampling |
| `sampling.top_p` | float | 0.95 | Top-p sampling |
| `sampling.min_p` | float | 0.05 | Min-p sampling |

### mtmd_sampling_params_t

Key sampling parameters:
- `seed`: Random seed (LLAMA_DEFAULT_SEED = 0xFFFFFFFF)
- `top_k`: Top-k sampling (<= 0 = vocab size)
- `top_p`: Top-p sampling (1.0 = disabled)
- `min_p`: Min-p sampling (0.0 = disabled)
- `temp`: Temperature (<= 0.0 = greedy)
- `penalty_last_n`: Last n tokens to penalize
- `penalty_repeat`: Repetition penalty (1.0 = disabled)
- `penalty_freq`: Frequency penalty (0.0 = disabled)
- `penalty_present`: Presence penalty (0.0 = disabled)

## Error Codes

| Code | Value | Description |
|------|-------|-------------|
| `MTMD_API_SUCCESS` | 0 | Success |
| `MTMD_API_ERROR_GENERIC` | -1 | Generic error |
| `MTMD_API_ERROR_INVALID_PARAM` | -2 | Invalid parameter |
| `MTMD_API_ERROR_MODEL_LOAD` | -3 | Failed to load model |
| `MTMD_API_ERROR_BACKEND_INIT` | -4 | Failed to initialize backend |
| `MTMD_API_ERROR_IMAGE_LOAD` | -5 | Failed to load image |
| `MTMD_API_ERROR_INFERENCE` | -6 | Inference error |
| `MTMD_API_ERROR_NOT_INITIALIZED` | -7 | Context not initialized |

## Integration with Existing Code

### Using the Library in Your CMake Project

```cmake
# Add the library
add_subdirectory(${LLAMA_CPP_DIR}/tools/mtmd-inference-api)

# Link against it
target_link_libraries(your_target PRIVATE mtmd-inference-api)
target_include_directories(your_target PRIVATE ${LLAMA_CPP_DIR}/tools/mtmd-inference-api)
```

### Example: Android JNI Integration

```cpp
// In your JNI native method
#include "mtmd-inference-api.h"

static mtmd_inference_context_t* g_inference_ctx = nullptr;

extern "C" JNIEXPORT jstring
JNICALL Java_com_misar_vlmanalyze_LocalInferenceBackend_analyzeImage(
    JNIEnv* env, jobject obj,
    jbyteArray jImageData, jint width, jint height,
    jstring jSystemPrompt, jstring jUserPrompt)
{
    // Initialize if not already
    if (!g_inference_ctx) {
        mtmd_inference_params_t params;
        mtmd_inference_params_default(&params);
        // ... configure params ...
        mtmd_inference_init(params, &g_inference_ctx, nullptr, nullptr);
    }

    // Convert image data
    jbyte* imageData = env->GetByteArrayElements(jImageData, nullptr);
    mtmd_image_data_t image;
    image.width = width;
    image.height = height;
    image.data = (const uint8_t*)imageData;

    // Prepare prompts
    mtmd_prompt_params_t prompt;
    // ... set prompts ...

    // Run inference
    mtmd_inference_result_t result;
    mtmd_inference_run(g_inference_ctx, &image, 1, prompt, &result, nullptr, nullptr);

    // Return result
    jstring resultStr = env->NewStringUTF(result.text.c_str());

    // Cleanup
    env->ReleaseByteArrayElements(jImageData, imageData, JNI_ABORT);
    return resultStr;
}
```

## License

This code is part of llama.cpp and follows the same license.
