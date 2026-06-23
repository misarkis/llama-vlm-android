// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// VLM-Analyze Android Implementation
// Integrates existing CPP source: api.cpp, encode.cpp, config.cpp

#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>

// Include existing CPP headers
#include "config.hpp"
#include "encode.hpp"
#include "api.hpp"
#include "json_parser.hpp"

// Include shared debug flags header
#include "debug-flags.h"

#define LOG_TAG "VLM-Impl"
#define LOGI(...) do { if (g_info_logs_enabled) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while(0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) do { if (g_debug_logs_enabled) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while(0)

// Global asset manager for loading models/config
AAssetManager* g_asset_manager = nullptr;  // Exposed for native-lib.cpp
static std::string g_storage_path;
static bool g_initialized = false;
static int64_t g_last_capture_time = 0;

using namespace screen_vlm;

/**
 * Load config.json from Android assets
 */
static std::string load_config_from_assets() {
    if (!g_asset_manager) {
        LOGE("Asset manager not available");
        return "";
    }

    AAsset* asset = AAssetManager_open(g_asset_manager, "config.json", AASSET_MODE_STREAMING);
    if (!asset) {
        LOGE("Failed to open config.json from assets");
        return "";
    }

    std::stringstream buffer;
    const void* data = AAsset_getBuffer(asset);
    if (data) {
        size_t length = AAsset_getLength(asset);
        buffer.write(static_cast<const char*>(data), length);
    }

    AAsset_close(asset);
    LOGI("Loaded config.json from assets (%zu bytes)", buffer.str().length());
    return buffer.str();
}

/**
 * Convert RGBA buffer to RGB ImageData format
 */
static ImageData rgba_to_imagedata(const uint8_t* rgba, int width, int height, int row_stride, int pixel_stride) {
    ImageData img;
    img.width = width;
    img.height = height;
    img.pixels.resize(width * height * 3);

    const uint8_t* src = rgba;
    uint8_t* dst = img.pixels.data();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst[0] = src[0];  // R
            dst[1] = src[1];  // G
            dst[2] = src[2];  // B
            src += pixel_stride;
            dst += 3;
        }
        // Skip padding at end of row
        src += row_stride - (width * pixel_stride);
    }

    return img;
}

/**
 * Initialize the VLM analyzer.
 * Called from Java when activity starts.
 */
extern "C" int vlmanalyze_init(const char* storage_path) {
    LOGI("vlmanalyze_init called with path: %s", storage_path);

    if (g_initialized) {
        LOGI("VLM already initialized");
        return 0;
    }

    g_storage_path = storage_path;

    try {
        // First try to load config from assets (Android-specific)
        std::string config_json = load_config_from_assets();

        if (!config_json.empty()) {
            // Parse and apply config from assets
            try {
                auto j = json::json::parse(config_json);
                if (j.contains("model")) Config::s_model = j["model"].get<std::string>();
                if (j.contains("api_url")) Config::s_api_url = j["api_url"].get<std::string>();
                if (j.contains("system_prompt")) Config::s_system_prompt = j["system_prompt"].get<std::string>();
                if (j.contains("user_prompt")) Config::s_user_prompt = j["user_prompt"].get<std::string>();
                if (j.contains("capture_cooldown")) Config::s_capture_cooldown = j["capture_cooldown"].get<int32_t>();
                if (j.contains("max_tokens")) Config::s_max_tokens = j["max_tokens"].get<int32_t>();
                if (j.contains("temperature")) Config::s_temperature = j["temperature"].get<double>();
                if (j.contains("top_p")) Config::s_top_p = j["top_p"].get<double>();
                if (j.contains("image_quality")) Config::s_image_quality = j["image_quality"].get<int32_t>();
                if (j.contains("image_scale")) Config::s_image_scale = j["image_scale"].get<int32_t>();
                if (j.contains("tts_rate_analysis")) Config::s_tts_rate_analysis = j["tts_rate_analysis"].get<int32_t>();
                if (j.contains("tts_rate_result")) Config::s_tts_rate_result = j["tts_rate_result"].get<int32_t>();
                LOGI("Config loaded from assets");
            } catch (const std::exception& e) {
                LOGE("Failed to parse config.json: %s", e.what());
                // Will use defaults below
            }
        } else {
            // Fall back to file-based config (for testing)
            Config::initialize(storage_path);
        }

        LOGI("VLM initialized successfully");
        LOGI("  API URL: %s", Config::get_api_url().c_str());
        LOGI("  Model: %s", Config::get_model().c_str());
        LOGI("  Max tokens: %d", Config::get_max_tokens());

        g_initialized = true;
        return 0;

    } catch (const std::exception& e) {
        LOGE("Failed to initialize VLM: %s", e.what());
        return -1;
    }
}

/**
 * Process a screen frame and get VLM analysis.
 * Full pipeline: capture -> resize -> encode -> API -> result
 */
extern "C" int vlmanalyze_process_frame(
    const uint8_t* data,
    int width, int height,
    int row_stride, int pixel_stride,
    char* result, int result_size) {

    if (!g_initialized) {
        LOGE("VLM not initialized");
        snprintf(result, result_size, "Error: VLM not initialized");
        return -1;
    }

    // Check cooldown (3 seconds default)
    auto now = std::chrono::steady_clock::now();
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    if (now_ms - g_last_capture_time < Config::get_capture_cooldown() * 1000LL) {
        // Still in cooldown - skip this frame
        return 0;
    }
    g_last_capture_time = now_ms;

    LOGI("Processing frame: %dx%d", width, height);

    try {
        // Step 1: Convert RGBA to RGB ImageData
        ImageData img = rgba_to_imagedata(data, width, height, row_stride, pixel_stride);

        // Step 2: Resize and encode to base64 JPEG
        std::string base64 = ImageEncoder::image_to_base64(img);
        if (base64.empty()) {
            LOGE("Failed to encode image");
            snprintf(result, result_size, "Error: Image encoding failed");
            return -1;
        }
        LOGI("Base64 length: %zu", base64.length());

        // Step 3: Send to VLM API
        auto start = std::chrono::high_resolution_clock::now();

        ApiResponse response = VlmApi::analyze_image(
            base64,
            Config::get_model(),
            Config::get_system_prompt(),
            Config::get_user_prompt(),
            [](const std::string& chunk, int32_t total_tokens) {
                // Streaming callback - could update UI here
                LOGI("Chunk: %s (%d tokens)", chunk.c_str(), total_tokens);
            }
        );

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();

        if (response.success) {
            LOGI("Analysis complete in %.2fs", duration);
            LOGI("Tokens: prompt=%d completion=%d", response.prompt_tokens, response.completion_tokens);

            // Copy result to output buffer
            std::string full_result = response.content;
            int len = std::min(static_cast<int>(full_result.length()), result_size - 1);
            memcpy(result, full_result.c_str(), len);
            result[len] = '\0';

            return len;
        } else {
            LOGE("API error: %s", response.error_message.c_str());
            snprintf(result, result_size, "Error: %s", response.error_message.c_str());
            return -1;
        }

    } catch (const std::exception& e) {
        LOGE("Processing error: %s", e.what());
        snprintf(result, result_size, "Error: %s", e.what());
        return -1;
    }
}

/**
 * Cleanup and destroy the VLM analyzer.
 */
extern "C" void vlmanalyze_destroy() {
    LOGI("vlmanalyze_destroy called");
    g_initialized = false;
    g_last_capture_time = 0;
}
