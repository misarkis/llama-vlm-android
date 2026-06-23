// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// test.cpp
// Test program for mtmd-inference-api
// Runs on Android device to verify the API works correctly
//
// Usage on device:
//   export LD_LIBRARY_PATH=/data/local/tmp
//   ./mtmd-inference-test \
//     --model /data/local/tmp/models/Qwen2-VL-2B.gguf \
//     --mmproj /data/local/tmp/models/mmproj-f16.gguf \
//     --image /data/local/tmp/test.png \
//     --system "You are a helpful assistant." \
//     --user "What is in this image?"
//
// Or with bitmap test (generates a test pattern):
//   ./mtmd-inference-test --bitmap-test --model ... --mmproj ...

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "mtmd-inference-api.h"
#include "arg.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

// Set environment variables for Android backend loading
static void set_android_env_vars(const std::string& device) {
    // LD_LIBRARY_PATH: current dir for finding .so files
    setenv("LD_LIBRARY_PATH", ".:./vendor/lib64", 1);
    // ADSP_LIBRARY_PATH: needed for Hexagon backend
    setenv("ADSP_LIBRARY_PATH", ".", 1);
    // GGML_HEXAGON_NDEV: only set to 4 for NPU (HTP), otherwise 0
    if (device.find("HTP") != std::string::npos) {
        setenv("GGML_HEXAGON_NDEV", "1", 1);
    } else {
        setenv("GGML_HEXAGON_NDEV", "0", 1);
    }
}

// Test image storage - holds image data for bitmap test
static std::vector<uint8_t> g_test_image_data;

// Test image generation - creates a simple color pattern
static bool generate_test_bitmap(mtmd_image_data_t& img, int width, int height) {
    img.width = width;
    img.height = height;

    g_test_image_data.resize(width * height * 3);
    img.data = g_test_image_data.data();

    // Create a simple gradient pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (size_t)(y * width + x) * 3;
            g_test_image_data[idx + 0] = (uint8_t)((x * 255) / width);      // R: horizontal gradient
            g_test_image_data[idx + 1] = (uint8_t)((y * 255) / height);     // G: vertical gradient
            g_test_image_data[idx + 2] = (uint8_t)(((x + y) * 127) / (width + height));  // B: diagonal
        }
    }

    return true;
}

static void free_test_bitmap(mtmd_image_data_t& img) {
    g_test_image_data.clear();
    img.data = nullptr;
}

// Progress callback
static void on_progress(const char* message, void* user_data) {
    printf("[Progress] %s\n", message);
}

// Token callback for streaming
static void on_token(const char* token, void* user_data) {
    printf("%s", token);
    fflush(stdout);
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>        Path to model GGUF file (required)\n");
    printf("  --mmproj <path>       Path to mmproj GGUF file (required)\n");
    printf("  --image <path>        Path to test image (optional, use with --bitmap-test)\n");
    printf("  --bitmap-test         Use generated test bitmap instead of file\n");
    printf("  --system <text>       System prompt (optional)\n");
    printf("  --user <text>         User prompt (required if not --bitmap-test)\n");
    printf("  --device <dev>        Device: CPU, GPU0, HTP0,HTP1,HTP2,HTP3 (default: CPU)\n");
    printf("  --n-gpu-layers <n>    GPU layers (0=CPU, 999=all, default: 999)\n");
    printf("  --n-ctx <n>           Context size (default: 16384)\n");
    printf("  --n-predict <n>       Max tokens to generate (default: 128)\n");
    printf("  --temp <f>            Temperature (default: 0.8)\n");
    printf("  --help                Show this help\n");
}

int main(int argc, char** argv) {
    printf("=== mtmd-inference-api Test ===\n\n");

    // Parse arguments
    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string system_prompt;
    std::string user_prompt;
    std::string device = "CPU";
    bool bitmap_test = false;
    int n_gpu_layers = 999;
    int n_ctx = 16384;
    int n_predict = 128;
    float temp = 0.8f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (arg == "--user" && i + 1 < argc) {
            user_prompt = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (arg == "--n-predict" && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = atof(argv[++i]);
        } else if (arg == "--bitmap-test") {
            bitmap_test = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate arguments
    if (model_path.empty() || mmproj_path.empty()) {
        fprintf(stderr, "Error: --model and --mmproj are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!bitmap_test && user_prompt.empty()) {
        fprintf(stderr, "Error: --user prompt is required (or use --bitmap-test)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Set Android environment variables for backend loading (after device is known)
    set_android_env_vars(device);

    // Step 1: Initialize parameters
    printf("Step 1: Initializing parameters...\n");
    mtmd_inference_params_t params;
    mtmd_inference_params_default(&params);

    params.model.path = model_path;
    params.model.mmproj_path = mmproj_path;
    params.device = device;
    params.n_gpu_layers = n_gpu_layers;
    params.n_ctx = n_ctx;
    params.n_predict = n_predict;
    params.sampling.temp = temp;
	params.verbose_prompt = true;
	params.use_mmap = true;

    printf("  Model: %s\n", params.model.path.c_str());
    printf("  MMProj: %s\n", params.model.mmproj_path.c_str());
    printf("  Device: %s\n", params.device.c_str());
    printf("  GPU Layers: %d\n", params.n_gpu_layers);
    printf("  Context: %d\n", params.n_ctx);
    printf("  Max Tokens: %d\n", params.n_predict);
    printf("  Temperature: %.2f\n", params.sampling.temp);
    printf("\n");

    // Step 2: Initialize the inference engine
    printf("Step 2: Initializing inference engine...\n");

    mtmd_inference_context_t* ctx = nullptr;
    mtmd_api_status status = mtmd_inference_init(
        params,
        &ctx,
        on_progress,
        nullptr
    );

    if (status != MTMD_API_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize: %s\n", mtmd_api_status_string(status));
        return 1;
    }

    printf("Initialization successful!\n\n");

    // Step 3: Prepare image data
    printf("Step 3: Preparing image data...\n");

    mtmd_image_data_t image;
    bool owns_image = false;

    if (bitmap_test) {
        // Generate test bitmap
        printf("  Generating test bitmap (256x256 gradient)...\n");
        owns_image = true;
        if (!generate_test_bitmap(image, 256, 256)) {
            fprintf(stderr, "Error: Failed to generate test bitmap\n");
            mtmd_inference_free(ctx);
            return 1;
        }
    } else {
        // Load image from file using stb_image
        printf("  Loading image from: %s\n", image_path.c_str());

        // Use stb_image to load the file
        int width, height, channels;
        unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);

        if (!data) {
            fprintf(stderr, "Error: Failed to load image: %s\n", stbi_failure_reason());
            mtmd_inference_free(ctx);
            return 1;
        }

        owns_image = true;
        image.width = width;
        image.height = height;
        image.data = data;

        printf("  Image loaded: %dx%d, channels: %d\n", width, height, channels);
    }

    printf("\n");

    // Step 4: Prepare prompt
    printf("Step 4: Preparing prompt...\n");

    mtmd_prompt_params_t prompt;
    prompt.system_prompt = system_prompt;
    prompt.user_prompt = user_prompt;
    prompt.add_ass_prefix = true;

    if (!prompt.system_prompt.empty()) {
        printf("  System: %s\n", prompt.system_prompt.c_str());
    }
    printf("  User: %s\n", prompt.user_prompt.c_str());
    printf("\n");

    // Step 5: Run inference
    printf("Step 5: Running inference...\n");
    printf("========================================\n");

    mtmd_inference_result_t result;

    status = mtmd_inference_run(
        ctx,
        &image,
        1,  // number of images
        prompt,
        &result,
        on_token,
        nullptr
    );

    printf("\n========================================\n\n");

    if (status != MTMD_API_SUCCESS) {
        fprintf(stderr, "Error: Inference failed: %s\n", mtmd_api_status_string(status));
        if (owns_image) free_test_bitmap(image);
        mtmd_inference_free(ctx);
        return 1;
    }

    // Step 6: Print results
    printf("=== Results ===\n");
    printf("\nFull response:\n%s\n\n", result.text.c_str());
    printf("Tokens generated: %d\n", result.n_tokens);
    printf("Prompt eval time: %.0f ms\n", result.prompt_eval_time_ms);
    printf("Generation time: %.0f ms\n", result.eval_time_ms);
    printf("Throughput: %.2f tokens/sec\n", result.tokens_per_second);
    printf("\n");

    // Step 7: Run another inference with same context (model already loaded)
    printf("Step 7: Running another inference with same context...\n");
    printf("========================================\n");

    mtmd_prompt_params_t prompt2;
    prompt2.system_prompt = "";  // No system prompt for follow-up
    prompt2.user_prompt = "Tell me more about what you see.";
    prompt2.add_ass_prefix = true;

    status = mtmd_inference_run(
        ctx,
        &image,
        1,
        prompt2,
        &result,
        on_token,
        nullptr
    );

    printf("\n========================================\n\n");

    if (status == MTMD_API_SUCCESS) {
        printf("Second response:\n%s\n\n", result.text.c_str());
        printf("Tokens generated: %d\n", result.n_tokens);
        printf("Throughput: %.2f tokens/sec\n", result.tokens_per_second);
    } else {
        fprintf(stderr, "Error: Second inference failed: %s\n", mtmd_api_status_string(status));
    }

    // Cleanup
    printf("\nStep 8: Cleaning up...\n");
    if (owns_image) {
        free_test_bitmap(image);
    }
    mtmd_inference_free(ctx);

    printf("Done.\n");
    return 0;
}
