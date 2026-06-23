// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// Android Text-to-Speech using OpenSL ES / AudioTrack
// Alternative: Use Android Java TTS via JNI (simpler, more reliable)

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#define LOG_TAG "VLM-TTS"

// Shared debug flags (defined in native-lib.cpp)
#include "debug-flags.h"
#define LOGI(...) do { if (g_info_logs_enabled) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while(0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Android TTS using Java TextToSpeech class via JNI
// This is simpler and more reliable than native OpenSL ES

static JavaVM* g_tts_jvm = nullptr;
static jobject g_tts_handler = nullptr;  // Reference to Java TTS helper class
static jmethodID g_ttsSpeakMid = nullptr;

/**
 * Initialize Android TTS
 * Called from Java side
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_TtsHelper_nativeInit(JNIEnv* env, jobject thiz) {
    LOGI("TTS nativeInit called");

    env->GetJavaVM(&g_tts_jvm);
    g_tts_handler = env->NewGlobalRef(thiz);

    jclass cls = env->GetObjectClass(thiz);
    g_ttsSpeakMid = env->GetMethodID(cls, "onTtsReady", "()V");
}

/**
 * Speak text using Android Java TTS
 * Called from Java TextToSpeech.OnInitListener
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_TtsHelper_speak(
    JNIEnv* env, jobject thiz, jstring text, jint rate) {

    if (!text) {
        LOGE("Null text provided to TTS");
        return;
    }

    const char* textC = env->GetStringUTFChars(text, nullptr);
    LOGI("TTS speaking: %s (rate=%d)", textC, rate);

    // The actual speech is handled by Java TextToSpeech class
    // This native method is just a logging hook

    env->ReleaseStringUTFChars(text, textC);
}

/**
 * Cleanup TTS resources
 */
extern "C" JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_TtsHelper_nativeDestroy(
    JNIEnv* env, jobject thiz) {

    LOGI("TTS nativeDestroy called");

    if (g_tts_handler) {
        env->DeleteGlobalRef(g_tts_handler);
        g_tts_handler = nullptr;
    }
}
