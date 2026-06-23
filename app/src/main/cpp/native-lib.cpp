// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstring>
#include <string>
#include <vector>

// Include existing CPP headers
#include "config.hpp"
#include "encode.hpp"
#include "api.hpp"

// Include shared debug flags header (define g_debug_logs_enabled here)
#include "debug-flags.h"

// Forward declare global asset manager in vlm-impl.cpp
extern AAssetManager* g_asset_manager;

// Android NDK screen capture and VLM analysis native implementation

#define LOG_TAG "VLM-Native"
#define LOGI(...) do { if (g_info_logs_enabled) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while(0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) do { if (g_debug_logs_enabled) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while(0)

// Global references
static JavaVM* g_jvm = nullptr;
static jobject g_activity = nullptr;       // MainActivity reference
// MainActivity callback method IDs
static jmethodID g_onAnalysisResultMid = nullptr;
static jmethodID g_onAnalysisErrorMid = nullptr;

// Debug logs enabled flag - set by Kotlin via JNI
// Defined here (declared in debug-flags.h)
bool g_debug_logs_enabled = false;

// Info logs enabled flag - set by Kotlin via JNI
bool g_info_logs_enabled = false;

// Forward declarations for VLM components (implemented in vlm-impl.cpp)
extern "C" int vlmanalyze_init(const char* storage_path);
extern "C" int vlmanalyze_process_frame(
    const uint8_t* data, int width, int height,
    int row_stride, int pixel_stride,
    char* result, int result_size);
extern "C" void vlmanalyze_destroy();

using namespace screen_vlm;

/**
 * JNI: Initialize the VLM analyzer
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_MainActivity_nativeInit(
    JNIEnv* env, jobject thiz, jobject assetManager, jstring storagePath) {

    LOGI("nativeInit called");

    // Get JavaVM reference
    env->GetJavaVM(&g_jvm);

    // Store global reference to activity
    g_activity = env->NewGlobalRef(thiz);

    // Get storage path
    const char* storagePathC = env->GetStringUTFChars(storagePath, nullptr);
    std::string storagePathStr(storagePathC);
    env->ReleaseStringUTFChars(storagePath, storagePathC);

    // Store asset manager for config loading
    g_asset_manager = AAssetManager_fromJava(env, assetManager);
    if (g_asset_manager) {
        LOGI("Asset manager stored successfully");
    } else {
        LOGE("Failed to get asset manager");
    }

    // Get callback method IDs
    jclass activityClass = env->GetObjectClass(thiz);
    g_onAnalysisResultMid = env->GetMethodID(activityClass, "onAnalysisResult", "(Ljava/lang/String;IIJDLjava/lang/String;)V");
    g_onAnalysisErrorMid = env->GetMethodID(activityClass, "onAnalysisError", "(Ljava/lang/String;)V");

    // Initialize native VLM
    int result = vlmanalyze_init(storagePathStr.c_str());
    if (result != 0) {
        LOGE("Failed to initialize VLM: %d", result);
        jstring errorMsg = env->NewStringUTF("Failed to initialize VLM");
        env->CallVoidMethod(thiz, g_onAnalysisErrorMid, errorMsg);
    } else {
        LOGI("VLM initialized successfully");
    }
}

/**
 * JNI: Process a screen frame (MainActivity)
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_MainActivity_nativeProcessFrame(
    JNIEnv* env, jobject thiz, jobject directBuffer, jint width, jint height, jint rowStride, jint pixelStride) {

    if (!directBuffer) {
        LOGE("Null buffer received");
        return;
    }

    // Get byte buffer address
    uint8_t* pixels = (uint8_t*)env->GetDirectBufferAddress(directBuffer);
    if (!pixels) {
        LOGE("Failed to get buffer address");
        return;
    }

    // Process frame with VLM
    char result[8192];
    int resultLen = vlmanalyze_process_frame(
        pixels, width, height, rowStride, pixelStride,
        result, sizeof(result));

    if (resultLen > 0) {
        // Callback to Java with result
        jstring resultJString = env->NewStringUTF(result);
        // Pass all parameters: result, promptTokens=0, completionTokens=0, totalTimeMs=0, tokensPerSecond=0.0, imagePath=null
        env->CallVoidMethod(thiz, g_onAnalysisResultMid, resultJString, 0, 0, (jlong)0, (jdouble)0.0, nullptr);
    } else if (resultLen < 0) {
        // Error occurred
        jstring errorMsg = env->NewStringUTF(result);
        env->CallVoidMethod(thiz, g_onAnalysisErrorMid, errorMsg);
    }
    // resultLen == 0 means skipped (cooldown) - no callback needed
}

/**
 * JNI: Cleanup on destroy (MainActivity)
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_MainActivity_nativeDestroy(
    JNIEnv* env, jobject thiz) {

    LOGI("nativeDestroy called (MainActivity)");
    vlmanalyze_destroy();

    if (g_activity) {
        env->DeleteGlobalRef(g_activity);
        g_activity = nullptr;
    }
    // Reset method IDs
    g_onAnalysisResultMid = nullptr;
    g_onAnalysisErrorMid = nullptr;
}

/**
 * JNI: Set debug logs enabled flag
 * Called from Kotlin when user toggles the debug logs button
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_MainActivity_nativeSetDebugLogsEnabled(
    JNIEnv* env, jobject thiz, jboolean enabled) {

    g_debug_logs_enabled = (enabled == JNI_TRUE);
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "nativeSetDebugLogsEnabled called: %s", g_debug_logs_enabled ? "true" : "false");
}

/**
 * JNI: Set info logs enabled flag
 * Called from Kotlin when user toggles the info logs button
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_MainActivity_nativeSetInfoLogsEnabled(
    JNIEnv* env, jobject thiz, jboolean enabled) {

    g_info_logs_enabled = (enabled == JNI_TRUE);
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "nativeSetInfoLogsEnabled called: %s", g_info_logs_enabled ? "true" : "false");
}

/**
 * JNI_OnLoad - Called when library is loaded
 */
extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;

    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    g_jvm = vm;
    LOGI("VLM Native library loaded");

    return JNI_VERSION_1_6;
}

/**
 * JNI_OnUnload - Called when library is unloaded
 */
extern "C" void JNICALL
JNI_OnUnload(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;

    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return;
    }

    LOGI("VLM Native library unloaded");
    vlmanalyze_destroy();
}
