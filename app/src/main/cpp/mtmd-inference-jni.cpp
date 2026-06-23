// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// mtmd-inference-jni.cpp
// JNI wrapper for mtmd-inference-api
// Provides direct multimodal inference without HTTP server

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <iostream>
#include <sstream>
#include <chrono>
#include <limits.h>

// Android logging
#define LOG_TAG "mtmd-jni"

// Shared debug flags (defined in native-lib.cpp)
#include "debug-flags.h"

#define LOGI(...) do { if (g_info_logs_enabled) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while(0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) do { if (g_debug_logs_enabled) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while(0)

// Include mtmd-inference-api
#include "mtmd-inference-api.h"
#include "llama.h"

// Include stb_image for image loading (from llama.cpp vendor)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


// ============================================================================
// Inference Context Wrapper
// ============================================================================
struct InferenceContext {
    mtmd_inference_context_t* ctx;
    bool valid;

    InferenceContext() : ctx(nullptr), valid(false) {}
};

// ============================================================================
// Global State
// ============================================================================
// Note: Backend initialization is now handled inside mtmd_inference_init()
// after environment variables are set. No global backend state needed here.

// ============================================================================
// Last Inference Result (for Java access)
// ============================================================================
static std::atomic<int32_t> g_last_prompt_tokens{0};
static std::atomic<int32_t> g_last_completion_tokens{0};
static std::atomic<int64_t> g_last_total_time_ms{0};
static std::atomic<double> g_last_tokens_per_second{0.0};

// ============================================================================
// Java String Helper
// ============================================================================
static std::string jstring_to_string(JNIEnv* env, jstring str) {
    if (!str) return "";
    const char* cstr = env->GetStringUTFChars(str, nullptr);
    std::string result(cstr);
    env->ReleaseStringUTFChars(str, cstr);
    return result;
}

static jstring string_to_jstring(JNIEnv* env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

// ============================================================================
// Get library directory from ApplicationContext
// ============================================================================
static std::string get_native_lib_dir(JNIEnv* env, jobject thiz) {
    std::string lib_dir;

    // Get ApplicationContext using a more reliable approach
    jclass thiz_class = env->GetObjectClass(thiz);
    if (!thiz_class) {
        LOGE("Failed to get class of thiz");
        return lib_dir;
    }

    // Use GetMethodID with the exact signature
    jmethodID get_app_ctx = env->GetMethodID(thiz_class, "getApplicationContext", "()Landroid/content/Context;");
    if (!get_app_ctx) {
        // Try parent class
        jclass activity_class = env->FindClass("android/app/Activity");
        if (activity_class) {
            get_app_ctx = env->GetMethodID(activity_class, "getApplicationContext", "()Landroid/content/Context;");
            env->DeleteLocalRef(activity_class);
        }
    }

    if (!get_app_ctx) {
        LOGE("Failed to get getApplicationContext method ID - using fallback path");
        env->DeleteLocalRef(thiz_class);
        // Fallback: use hardcoded path based on Android app structure
        return "/data/app/*/lib/arm64";
    }

    jobject app_ctx = env->CallObjectMethod(thiz, get_app_ctx);
    env->DeleteLocalRef(thiz_class);
    if (!app_ctx) {
        LOGE("Failed to get ApplicationContext");
        return lib_dir;
    }

    // Get ApplicationInfo from Context
    jclass ctx_class = env->FindClass("android/content/Context");
    if (!ctx_class) {
        LOGE("Failed to find Context class");
        env->DeleteLocalRef(app_ctx);
        return lib_dir;
    }

    jmethodID get_app_info = env->GetMethodID(ctx_class, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
    env->DeleteLocalRef(ctx_class);
    if (!get_app_info) {
        LOGE("Failed to get getApplicationInfo method ID");
        env->DeleteLocalRef(app_ctx);
        return lib_dir;
    }

    jobject app_info = env->CallObjectMethod(app_ctx, get_app_info);
    env->DeleteLocalRef(app_ctx);
    if (!app_info) {
        LOGE("Failed to get ApplicationInfo");
        return lib_dir;
    }

    // Get nativeLibraryDir from ApplicationInfo
    jclass app_info_class = env->GetObjectClass(app_info);
    if (!app_info_class) {
        LOGE("Failed to get ApplicationInfo class");
        env->DeleteLocalRef(app_info);
        return lib_dir;
    }

    jfieldID native_lib_dir_fid = env->GetFieldID(app_info_class, "nativeLibraryDir", "Ljava/lang/String;");
    env->DeleteLocalRef(app_info_class);
    if (!native_lib_dir_fid) {
        LOGE("Failed to get nativeLibraryDir field ID");
        env->DeleteLocalRef(app_info);
        return lib_dir;
    }

    jstring native_lib_dir_jstr = (jstring)env->GetObjectField(app_info, native_lib_dir_fid);
    env->DeleteLocalRef(app_info);
    if (!native_lib_dir_jstr) {
        LOGE("Failed to get nativeLibraryDir string");
        return lib_dir;
    }

    const char* native_lib_dir = env->GetStringUTFChars(native_lib_dir_jstr, nullptr);
    if (native_lib_dir) {
        lib_dir = native_lib_dir;
        LOGI("Got native library directory: %s", native_lib_dir);
        env->ReleaseStringUTFChars(native_lib_dir_jstr, native_lib_dir);
    } else {
        LOGE("Failed to get UTF chars from nativeLibraryDir string");
    }

    env->DeleteLocalRef(native_lib_dir_jstr);

    return lib_dir;
}

// ============================================================================
// Progress Callback
// ============================================================================
static void on_progress(const char* message, void* user_data) {
    LOGD("[Progress] %s", message ? message : "null");
}

// ============================================================================
// Token Callback for Streaming
// ============================================================================
// Java callback method reference (global)
static JavaVM* g_jvm = nullptr;
static jclass g_callback_class = nullptr;
static jmethodID g_on_token_mid = nullptr;

// Streaming state tracking
static std::atomic<int> g_token_count{0};
static std::atomic<bool> g_streaming_active{false};
static std::atomic<bool> g_stop_generation{false};
static std::chrono::steady_clock::time_point g_stream_start_time;

// Shared token buffer for real-time streaming (thread-safe)
static std::string g_stream_buffer;
static std::mutex g_stream_buffer_mutex;

// Check if token contains garbage patterns that should stop generation
static bool contains_garbage_token(const std::string& token) {
    // Check for <|start_header_id|>, <|end_header_id|>, <|eot_id|>, etc.
	if (token.find("<|") != std::string::npos) return true;
	if (token.find("|>") != std::string::npos) return true;
    if (token.find("<|start_header_id|>") != std::string::npos) return true;
    if (token.find("<|end_header_id|>") != std::string::npos) return true;
    if (token.find("<|eot_id|>") != std::string::npos) return true;
    if (token.find("<|start_header|>") != std::string::npos) return true;
    if (token.find("<|end_header|>") != std::string::npos) return true;
    // Qwen image/video tokens
    if (token.find("<|box_end|>") != std::string::npos) return true;
    if (token.find("︎<|box_start|>") != std::string::npos) return true;
    return false;
}

static void on_token(const char* token, void* user_data) {
    if (!token) return;

    std::string token_str(token);

    // Increment token count
    g_token_count.fetch_add(1, std::memory_order_relaxed);
	
    // Check for garbage tokens - stop generation immediately
    if (contains_garbage_token(token_str)) {
        LOGI("[on_token] Garbage token detected, stopping generation: %s", token_str.c_str());
        g_stop_generation.store(true, std::memory_order_release);
        g_streaming_active.store(false, std::memory_order_release);
        return;  // Don't forward this token to Java
    }

    
	// Append token to shared buffer (thread-safe)
    {
        std::lock_guard<std::mutex> lock(g_stream_buffer_mutex);
        g_stream_buffer += token_str;
    }

    // Call back to Java if callback is registered
    if (g_jvm && g_callback_class && g_on_token_mid) {
        JNIEnv* env = nullptr;
        bool attach = false;

        // Get current thread's JNIEnv
        jint get_env_result = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
        if (get_env_result == JNI_EDETACHED) {
            if (g_jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
                return;
            }
            attach = true;
        }

        if (env) {
            jstring token_jstr = env->NewStringUTF(token_str.c_str());
            env->CallStaticVoidMethod(g_callback_class, g_on_token_mid, token_jstr);
            if (token_jstr) env->DeleteLocalRef(token_jstr);
        }

        if (attach) {
            g_jvm->DetachCurrentThread();
        }
    } else {
        // Fallback: just log
        LOGD("[Token] %s", token_str.c_str());
    }
}

// Get token count for streaming
extern "C" JNIEXPORT jint JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetTokenCount(JNIEnv* env, jobject thiz) {
    return g_token_count.load(std::memory_order_relaxed);
}

// Get current stream buffer contents (for polling)
extern "C" JNIEXPORT jstring JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetStreamBuffer(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_stream_buffer_mutex);
    return string_to_jstring(env, g_stream_buffer);
}

// Reset stream buffer (call before starting new inference)
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeResetStreamBuffer(JNIEnv* env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_stream_buffer_mutex);
    g_stream_buffer.clear();
}

// ============================================================================
// JNI Native Methods
// ============================================================================

extern "C" {

/**
 * Initialize the inference engine.
 *
 * @param modelPath    Path to model GGUF file
 * @param mmprojPath   Path to multimodal projector GGUF file
 * @param device       Device string: "CPU0", "GPU0", "HTP0,HTP1,HTP2,HTP3"
 * @param nGpuLayers   Number of layers to offload (0 = CPU, 999 = all GPU/NPU)
 * @param nCtx         Context size
 * @param nBatch       Batch size
 * @param nPredict     Max tokens to generate (default: 512)
 * @param verbose      Enable verbose logging (default: false)
 * @param imageMinTokens  Minimum image tokens (-1 = unlimited)
 * @param topK         Top-k sampling (<= 0 = vocab size)
 * @param topP         Top-p sampling (1.0 = disabled)
 * @param temperature  Sampling temperature (<= 0.0 = greedy)
 * @param repetitionPenalty  Repetition penalty (1.0 = disabled)
 * @param debugLogsEnabled Enable debug logging (default: false)
 * @param infoLogsEnabled Enable info logging (default: true)
 * @return             Opaque handle (InferenceContext*) or 0 on failure
 */
JNIEXPORT jlong JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeInit(
    JNIEnv* env, jobject thiz,
    jstring modelPath, jstring mmprojPath, jstring libDir,
    jstring device, jint nGpuLayers, jint nCtx, jint nBatch, jint nPredict, jboolean verbose,
    jint imageMinTokens, jint topK, jfloat topP, jfloat temperature, jfloat repetitionPenalty, jboolean debugLogsEnabled, jboolean infoLogsEnabled) {

    LOGI("=== nativeInit called ===");

    std::string model = jstring_to_string(env, modelPath);
    std::string mmproj = jstring_to_string(env, mmprojPath);
    std::string dev = jstring_to_string(env, device);

    if (model.empty() || mmproj.empty()) {
        LOGE("Model path and mmproj path are required");
        return 0;
    }

    if (dev.empty()) {
        dev = "CPU0";
    }

    LOGI("Model: %s", model.c_str());
    LOGI("MMProj: %s", mmproj.c_str());
    LOGI("Device: %s", dev.c_str());
    LOGI("GPU Layers: %d", nGpuLayers);
    LOGI("Context: %d", nCtx);
    LOGI("Batch: %d", nBatch);
    LOGI("Predict: %d", nPredict);
    LOGI("Verbose: %s", verbose ? "true" : "false");
    LOGI("Image Min Tokens: %d", imageMinTokens);
    LOGI("Top K: %d", topK);
    LOGI("Top P: %.2f", topP);
    LOGI("Temperature: %.2f", temperature);
    LOGI("Repetition Penalty: %.2f", repetitionPenalty);

    // Get library directory from Java parameter
    std::string lib_dir = jstring_to_string(env, libDir);
    if (lib_dir.empty()) {
        LOGE("Library directory is empty");
        return 0;
    }
    LOGI("Library directory: %s", lib_dir.c_str());

    // Initialize parameters
    mtmd_inference_params_t params;
    mtmd_inference_params_default(&params);

    params.model.path = model;
    params.model.mmproj_path = mmproj;
    params.model.mmproj_use_gpu = (nGpuLayers > 0);
    params.device = dev;
    params.library_directory = lib_dir;  // Pass lib_dir for env setup and backend loading
    params.n_gpu_layers = nGpuLayers;
    params.n_ctx = nCtx > 0 ? nCtx : 16384;
    params.n_batch = nBatch > 0 ? nBatch : 512;
    params.n_predict = nPredict > 0 ? nPredict : INT_MAX;
    params.cpuparams.n_threads = 4;
    params.verbose_prompt = verbose;
    params.debug_logs_enabled = debugLogsEnabled;
    params.info_logs_enabled = infoLogsEnabled;
    params.use_mmap = false;

    // Inference parameters
    params.image_min_tokens = imageMinTokens;
    params.sampling.top_k = topK;
    params.sampling.top_p = topP;
    params.sampling.temp = temperature;
    params.sampling.penalty_repeat = repetitionPenalty;

    // Create context
    InferenceContext* ctx = new InferenceContext();

    // Check if model file exists and is accessible
    FILE* test_file = fopen(model.c_str(), "rb");
    if (!test_file) {
        LOGE("Cannot open model file: %s", model.c_str());
        delete ctx;
        return 0;
    }
    fclose(test_file);
    LOGI("Model file is accessible");

    // Check if mmproj file exists and is accessible
    test_file = fopen(mmproj.c_str(), "rb");
    if (!test_file) {
        LOGE("Cannot open mmproj file: %s", mmproj.c_str());
        delete ctx;
        return 0;
    }
    fclose(test_file);
    LOGI("MMProj file is accessible");
	
	// Set LD_LIBRARY_PATH if library_directory is provided
    if (!params.library_directory.empty()) {
        std::string ld_lib_path = params.library_directory + ":/vendor/lib64:/system/lib64";
        setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(),1);
        LOGI("Set LD_LIBRARY_PATH=%s", ld_lib_path.c_str());

        // Set ADSP_LIBRARY_PATH to library_directory
		std::string adsp_lib_path = params.library_directory;
        setenv("ADSP_LIBRARY_PATH", adsp_lib_path.c_str(),1);
        LOGI("Set ADSP_LIBRARY_PATH=%s", adsp_lib_path.c_str());
    }
	
    mtmd_inference_context_t* mtmd_ctx = nullptr;
    mtmd_api_status status = mtmd_inference_init(
        params,
        &mtmd_ctx,
        on_progress,
        nullptr
    );

    LOGI("mtmd_inference_init returned: %s", mtmd_api_status_string(status));

    if (status != MTMD_API_SUCCESS) {
        LOGE("mtmd_inference_init failed: %s", mtmd_api_status_string(status));
        delete ctx;
        return 0;
    }

    ctx->ctx = mtmd_ctx;
    ctx->valid = true;

    LOGI("Initialization successful");
    return reinterpret_cast<jlong>(ctx);
}

// Forward declaration
static std::string run_inference(InferenceContext* ctx, mtmd_image_data_t* image,
                                  const std::string& user_prompt, int maxTokens,
                                  const std::string& system_prompt = "");

/**
 * Run inference on an image file.
 *
 * @param handle     Inference context handle
 * @param imagePath  Path to image file
 * @param prompt     User prompt (system prompt is empty)
 * @param maxTokens  Maximum tokens to generate
 * @return           Generated text
 */
JNIEXPORT jstring JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGenerate(
    JNIEnv* env, jobject thiz,
    jlong handle, jstring imagePath, jstring prompt, jint maxTokens) {

    if (handle == 0) {
        LOGE("Invalid handle");
        return env->NewStringUTF("Error: Invalid handle");
    }

    InferenceContext* ctx = reinterpret_cast<InferenceContext*>(handle);
    if (!ctx->valid || !ctx->ctx) {
        LOGE("Context not valid");
        return env->NewStringUTF("Error: Context not valid");
    }

    std::string img_path = jstring_to_string(env, imagePath);
    std::string user_prompt = jstring_to_string(env, prompt);

    if (img_path.empty()) {
        LOGE("Image path is required");
        return env->NewStringUTF("Error: Image path is required");
    }

    if (user_prompt.empty()) {
        user_prompt = "Describe this image.";
    }

    LOGI("Running inference: image=%s, prompt=%s", img_path.c_str(), user_prompt.c_str());
    LOGI("Max tokens: %d", maxTokens);

    // Load image from file using stb_image
    int width, height, channels;
    unsigned char* img_data = stbi_load(img_path.c_str(), &width, &height, &channels, 3);

    if (!img_data) {
        LOGE("Failed to load image: %s", img_path.c_str());
        std::string err_msg = std::string("Error: Failed to load image: ") + stbi_failure_reason();
        return string_to_jstring(env, err_msg);
    }

    LOGI("Image loaded: %dx%d, channels: %d", width, height, channels);

    // Prepare image data
    mtmd_image_data_t image;
    image.width = width;
    image.height = height;
    image.data = img_data;

    // Run inference (no system prompt for file-based inference)
    std::string result = run_inference(ctx, &image, user_prompt, maxTokens, "");

    // Free image data
    stbi_image_free(img_data);

    return string_to_jstring(env, result);
}

/**
 * Run inference on an image provided as raw RGB data in memory.
 *
 * @param handle       Inference context handle
 * @param imageData    Byte array containing RGB image data
 * @param width        Image width in pixels
 * @param height       Image height in pixels
 * @param prompt       User prompt
 * @param maxTokens    Maximum tokens to generate
 * @param systemPrompt System prompt (can be null)
 * @return             Generated text
 */
JNIEXPORT jstring JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGenerateFromBytes(
    JNIEnv* env, jobject thiz,
    jlong handle, jbyteArray imageData, jint width, jint height,
    jstring prompt, jint maxTokens, jstring systemPrompt) {

    if (handle == 0) {
        LOGE("Invalid handle");
        return env->NewStringUTF("Error: Invalid handle");
    }

    InferenceContext* ctx = reinterpret_cast<InferenceContext*>(handle);
    if (!ctx->valid || !ctx->ctx) {
        LOGE("Context not valid");
        return env->NewStringUTF("Error: Context not valid");
    }

    std::string user_prompt = jstring_to_string(env, prompt);
    if (user_prompt.empty()) {
        user_prompt = "Describe this image.";
    }

    std::string sys_prompt = systemPrompt ? jstring_to_string(env, systemPrompt) : "";

    LOGI("=== nativeGenerateFromBytes START ===");
    LOGI("Image: %dx%d = %d pixels, prompt=%s, system_prompt=%s", width, height, width * height, user_prompt.c_str(), sys_prompt.c_str());
    LOGI("Max tokens: %d", maxTokens);

    // Get byte array elements
    jsize img_size = width * height * 3;  // RGB format
    jbyte* img_bytes = env->GetByteArrayElements(imageData, nullptr);

    if (!img_bytes) {
        LOGE("Failed to get image byte array");
        return env->NewStringUTF("Error: Failed to get image data");
    }

    // Copy to unsigned char buffer for inference (we need our own copy since we'll release the Java array)
    unsigned char* img_data = (unsigned char*)malloc(img_size);
    if (!img_data) {
        env->ReleaseByteArrayElements(imageData, img_bytes, JNI_ABORT);
        LOGE("Failed to allocate image buffer");
        return env->NewStringUTF("Error: Failed to allocate image buffer");
    }
    memcpy(img_data, img_bytes, img_size);

    LOGI("Image data copied: %d bytes", img_size);

    // Prepare image data
    mtmd_image_data_t image;
    image.width = width;
    image.height = height;
    image.data = img_data;

    // Run inference
    std::string result = run_inference(ctx, &image, user_prompt, maxTokens, sys_prompt);

    // Free our copy and release Java array
    free(img_data);
    env->ReleaseByteArrayElements(imageData, img_bytes, JNI_ABORT);

    LOGI("=== nativeGenerateFromBytes DONE ===");
    return string_to_jstring(env, result);
}

/**
 * Helper function to run inference with an image.
 * Returns the generated text or an error message.
 */
static std::string run_inference(InferenceContext* ctx, mtmd_image_data_t* image,
                                  const std::string& user_prompt, int maxTokens,
                                  const std::string& system_prompt) {

    LOGI("[run_inference] Preparing prompt");

    // Reset streaming state
    g_token_count.store(0, std::memory_order_relaxed);
    g_streaming_active.store(true, std::memory_order_release);
    g_stop_generation.store(false, std::memory_order_release);
    g_stream_start_time = std::chrono::steady_clock::now();

    // Reset shared buffer
    {
        std::lock_guard<std::mutex> lock(g_stream_buffer_mutex);
        g_stream_buffer.clear();
    }

    // Prepare prompt
    mtmd_prompt_params_t prompt_params;
    prompt_params.system_prompt = system_prompt.empty() ? nullptr : system_prompt.c_str();
    prompt_params.user_prompt = user_prompt;
    prompt_params.add_ass_prefix = true;

    LOGI("[run_inference] Calling mtmd_inference_run");

    // Run inference
    mtmd_inference_result_t result;

    mtmd_api_status status = mtmd_inference_run(
        ctx->ctx,
        image,
        1,  // number of images
        prompt_params,
        &result,
        on_token,
        nullptr
    );

    g_streaming_active.store(false, std::memory_order_release);

    LOGI("[run_inference] mtmd_inference_run returned: %s", mtmd_api_status_string(status));

    if (status != MTMD_API_SUCCESS) {
        LOGE("Inference failed: %s", mtmd_api_status_string(status));
        return std::string("Error: Inference failed: ") + mtmd_api_status_string(status);
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - g_stream_start_time).count();

    // Store result in global atomics for Java access
    g_last_completion_tokens.store(result.n_tokens, std::memory_order_relaxed);
    g_last_prompt_tokens.store(result.n_prompt_tokens, std::memory_order_relaxed);
    g_last_total_time_ms.store(duration, std::memory_order_relaxed);
    g_last_tokens_per_second.store(result.tokens_per_second, std::memory_order_relaxed);

    LOGI("Inference complete: %d prompt + %d completion tokens in %lldms (%.2f t/s)",
         result.n_prompt_tokens, result.n_tokens, duration, result.tokens_per_second);

    // Return the shared buffer content (already filtered by on_token callback)
    // This avoids returning garbage tokens that may have been generated after detection
    std::string result_text;
    {
        std::lock_guard<std::mutex> lock(g_stream_buffer_mutex);
        result_text = g_stream_buffer;
    }

    if (result_text.empty()) {
        // Fallback to result.text if buffer is empty
        result_text = result.text;
    }

    LOGI("Returning %zu chars from stream buffer", result_text.length());
    return result_text;
}

/**
 * Free inference context.
 *
 * @param handle  Inference context handle
 */
JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeFree(
    JNIEnv* env, jobject thiz, jlong handle) {

    if (handle == 0) return;

    InferenceContext* ctx = reinterpret_cast<InferenceContext*>(handle);
    if (ctx->ctx) {
        LOGI("Freeing inference context");
        mtmd_inference_free(ctx->ctx);
    }
    delete ctx;

    LOGI("Context freed");
}

/**
 * Check if handle is valid.
 *
 * @param handle  Inference context handle
 * @return        true if valid, false otherwise
 */
JNIEXPORT jboolean JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeIsValid(
    JNIEnv* env, jobject thiz, jlong handle) {

    if (handle == 0) return JNI_FALSE;

    InferenceContext* ctx = reinterpret_cast<InferenceContext*>(handle);
    return ctx->valid && ctx->ctx ? JNI_TRUE : JNI_FALSE;
}

/**
 * Set token callback for streaming.
 * This is called from Java to register a callback class/method.
 */
JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeSetTokenCallback(
    JNIEnv* env, jobject thiz, jclass callbackClass, jmethodID methodId) {

    // Store global references
    if (g_callback_class) {
        env->DeleteGlobalRef(g_callback_class);
    }
    if (g_on_token_mid) {
        // Method ID is local to the JVM, no need to globalize
    }

    g_callback_class = (jclass)env->NewGlobalRef(callbackClass);
    g_on_token_mid = methodId;

    LOGI("Token callback registered");
}

/**
 * Get method ID for a method in a class.
 * Used for registering callback methods.
 */
JNIEXPORT jlong JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetMethodId(
    JNIEnv* env, jobject thiz, jclass callbackClass, jstring methodName) {

    if (!callbackClass || !methodName) {
        return 0;
    }

    std::string method_name = jstring_to_string(env, methodName);
    LOGI("Getting method ID for: %s", method_name.c_str());

    // Get the class of the callback object
    jclass clazz = env->GetObjectClass(callbackClass);
    if (!clazz) {
        LOGE("Failed to get class for callback");
        return 0;
    }

    // Get method ID for the specified method (Void return, String param)
    jmethodID mid = env->GetMethodID(clazz, method_name.c_str(), "(Ljava/lang/String;)V");
    if (!mid) {
        LOGE("Method not found: %s", method_name.c_str());
        env->DeleteLocalRef(clazz);
        return 0;
    }

    // Create a global reference to the class
    jclass global_class = (jclass)env->NewGlobalRef(clazz);
    env->DeleteLocalRef(clazz);

    // Return the method ID as a jlong (method IDs are stable within a JVM instance)
    return reinterpret_cast<jlong>(mid);
}

// ============================================================================
// JNI Getters for Last Inference Result
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetLastPromptTokens(
    JNIEnv* env, jobject thiz) {
    return g_last_prompt_tokens.load(std::memory_order_relaxed);
}

JNIEXPORT jint JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetLastCompletionTokens(
    JNIEnv* env, jobject thiz) {
    return g_last_completion_tokens.load(std::memory_order_relaxed);
}

JNIEXPORT jlong JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetLastTotalTimeMs(
    JNIEnv* env, jobject thiz) {
    return g_last_total_time_ms.load(std::memory_order_relaxed);
}

JNIEXPORT jdouble JNICALL
Java_com_misar_vlmanalyze_LocalInferenceService_nativeGetLastTokensPerSecond(
    JNIEnv* env, jobject thiz) {
    return g_last_tokens_per_second.load(std::memory_order_relaxed);
}

}  // extern "C"

// ============================================================================
// JNI OnLoad/OnUnload
// ============================================================================

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    LOGI("mtmd-inference-jni loaded");
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    if (g_callback_class) {
        JNIEnv* env = nullptr;
        if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) == JNI_OK) {
            env->DeleteGlobalRef(g_callback_class);
        }
        g_callback_class = nullptr;
    }
    g_on_token_mid = nullptr;
    LOGI("mtmd-inference-jni unloaded");
}
