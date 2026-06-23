// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#include <jni.h>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <iostream>
#include <sstream>
#include <dlfcn.h>
#include <exception>

// Android logging
#include <android/log.h>

#define TAG "VLM-JNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Shared debug flags (defined in native-lib.cpp)
#include "debug-flags.h"
#define LOGI(...) do { if (g_info_logs_enabled) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__); } while(0)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGD(...) do { if (g_debug_logs_enabled) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__); } while(0)

// Function pointer type for llama_server
typedef int (*llama_server_fn)(int argc, char **argv);

static void *g_impl_handle = nullptr;
static llama_server_fn g_llama_server_fn = nullptr;

static std::thread *g_server_thread = nullptr;
static std::atomic<bool> g_server_running(false);
static std::vector<std::string> g_server_args;  // Keep args alive while server runs

static std::string jstring_to_string(JNIEnv *env, jstring str) {
    const char *cstr = env->GetStringUTFChars(str, nullptr);
    std::string result(cstr);
    env->ReleaseStringUTFChars(str, cstr);
    return result;
}

// Convert std::vector<char*> to readable string for logging
static std::string argv_to_string(const std::vector<char *> &argv) {
    std::ostringstream oss;
    for (size_t i = 0; i < argv.size(); i++) {
        if (i > 0) oss << " ";
        oss << argv[i];
    }
    return oss.str();
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_misar_vlmanalyze_LlamaServerService_nativeStartServer(
        JNIEnv *env, jobject thiz,
        jstring modelPath, jstring mmprojPath,
        jint port, jint nCtx, jint nBatch,
        jint nGpuLayers, jstring device, jboolean verbose) {

    LOGI("=== nativeStartServer called ===");

    if (g_server_running.load()) {
        LOGE("Server already running");
        return JNI_FALSE;
    }

    std::string model = jstring_to_string(env, modelPath);
    std::string mmproj = jstring_to_string(env, mmprojPath);
    std::string dev = device ? jstring_to_string(env, device) : "";

    // Get the native library directory from ApplicationInfo (extracted location)
    // With extractNativeLibs="true", libraries are extracted to /data/app/.../lib/arm64-v8a/
    jclass serviceClass = env->GetObjectClass(thiz);

    // Get getApplicationContext method
    jmethodID getApplicationContextId = env->GetMethodID(serviceClass, "getApplicationContext", "()Landroid/content/Context;");
    jobject appContext = env->CallObjectMethod(thiz, getApplicationContextId);

    // Get getApplicationInfo method from Context
    jclass contextClass = env->FindClass("android/content/Context");
    jmethodID getApplicationInfoId = env->GetMethodID(contextClass, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
    jobject applicationInfo = env->CallObjectMethod(appContext, getApplicationInfoId);

    // Get ApplicationInfo class and nativeLibraryDir field (extracted .so location)
    jclass appInfoClass = env->FindClass("android/content/pm/ApplicationInfo");
    jfieldID nativeLibraryDirFieldId = env->GetFieldID(appInfoClass, "nativeLibraryDir", "Ljava/lang/String;");
    jstring nativeLibDirPath = (jstring)env->GetObjectField(applicationInfo, nativeLibraryDirFieldId);

    const char *nativeLibDir = env->GetStringUTFChars(nativeLibDirPath, nullptr);

    LOGI("Native library directory (extracted): %s", nativeLibDir);

    // Set LD_LIBRARY_PATH to extracted native lib dir + /vendor/lib64 + /system/lib64
    std::string ld_lib_path = std::string(nativeLibDir) + ":/vendor/lib64:/system/lib64";
    setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(), 1);
    LOGI("Set LD_LIBRARY_PATH=%s", ld_lib_path.c_str());

    // ADSP_LIBRARY_PATH points to the same extracted lib dir for Hexagon
    setenv("ADSP_LIBRARY_PATH", nativeLibDir, 1);
    LOGI("Set ADSP_LIBRARY_PATH=%s", nativeLibDir);

    env->ReleaseStringUTFChars(nativeLibDirPath, nativeLibDir);
    env->DeleteLocalRef(nativeLibDirPath);
    env->DeleteLocalRef(appInfoClass);
    env->DeleteLocalRef(applicationInfo);
    env->DeleteLocalRef(contextClass);
    env->DeleteLocalRef(appContext);
    env->DeleteLocalRef(serviceClass);

    // Set GGML_HEXAGON_NDEV based on backend
    // NPU: GGML_HEXAGON_NDEV=4, CPU/GPU: GGML_HEXAGON_NDEV=0
    if (!dev.empty() && dev.find("HTP") != std::string::npos) {
        setenv("GGML_HEXAGON_NDEV", "4", 1);
        LOGI("Set GGML_HEXAGON_NDEV=4 for NPU backend");
    } else {
        setenv("GGML_HEXAGON_NDEV", "0", 1);
        LOGI("Set GGML_HEXAGON_NDEV=0 for CPU/GPU backend");
    }

    // Pre-load libomp.so first (required for OpenMP support)
    LOGI("Pre-loading libomp.so...");
    void *omp_handle = dlopen("libomp.so", RTLD_NOW | RTLD_GLOBAL);
    if (omp_handle == nullptr) {
        LOGE("Failed to load libomp.so: %s", dlerror());
    } else {
        LOGI("libomp.so loaded successfully");
    }

    // Load libllama-server-impl.so dynamically with RTLD_GLOBAL to expose symbols
    if (g_impl_handle == nullptr) {
        LOGI("Loading libllama-server-impl.so dynamically...");
        g_impl_handle = dlopen("libllama-server-impl.so", RTLD_NOW | RTLD_GLOBAL);
        if (g_impl_handle == nullptr) {
            LOGE("dlopen failed: %s", dlerror());
            return JNI_FALSE;
        }
        LOGI("dlopen succeeded");

        // Use C++ mangled symbol name: _Z12llama_serveriPPc = llama_server(int, char**)
        g_llama_server_fn = (llama_server_fn)dlsym(g_impl_handle, "_Z12llama_serveriPPc");
        if (g_llama_server_fn == nullptr) {
            LOGE("dlsym failed for llama_server (mangled: _Z12llama_serveriPPc): %s", dlerror());
            dlclose(g_impl_handle);
            g_impl_handle = nullptr;
            return JNI_FALSE;
        }
        LOGI("dlsym succeeded - llama_server symbol found");
    }

    if (g_llama_server_fn == nullptr) {
        LOGE("llama_server function pointer is null");
        return JNI_FALSE;
    }

    LOGI("Model path: %s", model.c_str());
    LOGI("MMProj path: %s", mmproj.c_str());
    LOGI("Port: %d", port);
    LOGI("GPU layers: %d", nGpuLayers);
    LOGI("Device: %s", dev.empty() ? "null" : dev.c_str());
    LOGI("Verbose: %s", verbose ? "true" : "false");

    // Build command-line arguments (match working CLI from TEST_SERVER_MANUAL.md)
    g_server_args.clear();
    g_server_args.push_back("llama-server");

    // Device first (for NPU/HTP backend)
    if (!dev.empty()) {
        g_server_args.push_back("--device");
        g_server_args.push_back(dev);
    }

    // Model and mmproj
    g_server_args.push_back("-m");
    g_server_args.push_back(model);

    if (!mmproj.empty()) {
        g_server_args.push_back("--mmproj");
        g_server_args.push_back(mmproj);
    }

    // Host and port
    g_server_args.push_back("--host");
    g_server_args.push_back("127.0.0.1");
    g_server_args.push_back("--port");
    g_server_args.push_back(std::to_string(port));

    // Context and batch size
    g_server_args.push_back("-c");
    g_server_args.push_back(std::to_string(nCtx));
    g_server_args.push_back("-b");
    g_server_args.push_back(std::to_string(nBatch));

    // GPU layers
    g_server_args.push_back("-ngl");
    g_server_args.push_back(std::to_string(nGpuLayers));

    // Disable -v flag for now - it may cause issues
    // if (verbose) {
    //     g_server_args.push_back("-v");
    //     LOGI("Added -v flag for verbose logging");
    // }

    // Convert to char* array
    std::vector<char *> argv;
    for (auto &arg : g_server_args) {
        argv.push_back(arg.data());
    }

    LOGI("Command: %s", argv_to_string(argv).c_str());

    // Start server in background thread
    // Capture g_server_args (static) not argv (local vector that would dangle)
    g_server_running.store(true);
    g_server_thread = new std::thread([]() {
        LOGI("=== Server thread starting ===");
        // Build argv from static g_server_args to keep pointers valid
        std::vector<char *> argv;
        for (auto &arg : g_server_args) {
            argv.push_back(arg.data());
        }
        LOGI("About to call llama_server with %zu arguments", argv.size());
        int result = -1;
        try {
            LOGI("Calling llama_server...");
            result = g_llama_server_fn(argv.size(), argv.data());
            LOGI("llama_server returned normally");
        } catch (const std::exception& e) {
            LOGE("Server threw std::exception: %s", e.what());
            result = 1;
        } catch (...) {
            LOGE("Server threw unknown exception");
            result = 1;
        }
        LOGI("=== Server thread exited with code: %d ===", result);
        g_server_running.store(false);
    });

    LOGI("Server thread created, returning JNI_TRUE");
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_misar_vlmanalyze_LlamaServerService_nativeStopServer(
        JNIEnv *env, jobject thiz) {

    LOGI("nativeStopServer called");

    if (g_server_thread != nullptr && g_server_thread->joinable()) {
        g_server_thread->join();
        delete g_server_thread;
        g_server_thread = nullptr;
    }
    g_server_running.store(false);

    // Clean up dynamically loaded library
    if (g_impl_handle != nullptr) {
        dlclose(g_impl_handle);
        g_impl_handle = nullptr;
        g_llama_server_fn = nullptr;
        LOGI("dlclosed libllama-server-impl.so");
    }
}

JNIEXPORT jboolean JNICALL
Java_com_misar_vlmanalyze_LlamaServerService_nativeIsServerRunning(
        JNIEnv *env, jobject thiz) {
    return g_server_running.load() ? JNI_TRUE : JNI_FALSE;
}

}  // extern "C"
