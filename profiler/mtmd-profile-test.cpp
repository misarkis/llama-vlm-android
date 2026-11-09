// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// mtmd-profile-test.cpp
// Profiling test executable for Qwen2-VL-2B
// Loads images from a folder, runs inference with configurable settings,
// and outputs profiling data for analysis

#include "mtmd-inference-api.h"
#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "mtmd-profile-test"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) printf(__VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#define LOGD(...) printf(__VA_ARGS__)
#endif

// ============================================================================
// Settings File Structure
// ============================================================================

struct profile_test_settings {
    std::string model_path;
    std::string mmproj_path;
    std::string backend;           // "CPU", "GPU", "NPU-4Core", "NPU-1Core"
    std::string vit_device;        // vision encoder device ("GPUOpenCL", "HTP0", "CPU"); empty = default
    int32_t htp_sessions;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_predict;
    int32_t image_min_tokens;
    int32_t top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    std::string image_dir;
    int32_t htp_profile;           // 0 = off, 1 = basic, 2 = extended
    std::string system_prompt;
    std::string user_prompt;
    std::string output_csv;
    std::string output_log;
};

static void profile_test_settings_default(profile_test_settings* settings) {
    settings->model_path = "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B/Qwen2-VL-2B-Q4_0.gguf";
    settings->mmproj_path = "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B/mmproj-f16.gguf";
    settings->backend = "NPU-4Core";
    settings->vit_device = "";
    settings->htp_sessions = 4;
    settings->n_ctx = 16384;
    settings->n_batch = 512;
    settings->n_predict = 512;
    settings->image_min_tokens = 1000;
    settings->top_k = 40;
    settings->top_p = 0.9f;
    settings->temperature = 0.9f;
    settings->repetition_penalty = 1.0f;
    settings->image_dir = "/data/local/tmp/test_images/";
    settings->htp_profile = 1;
    settings->system_prompt = "You are a AI assistant. Please answer in details in English only. Just English. Do not use chineese. No chineese.";
    settings->user_prompt = "What is on this screen right now? What should the person see? Tell them what matters. Keep it casual like you are explaining it to a friend.";
    settings->output_csv = "/sdcard/profile_results.csv";
    settings->output_log = "/sdcard/profile_htp_log.txt";
}

// Simple JSON parser for settings file
static std::string json_get_string(const std::string& json, const std::string& key, const std::string& default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;

    pos = json.find("\"", pos);
    if (pos == std::string::npos) return default_val;

    size_t start = pos + 1;
    size_t end = json.find("\"", start);
    if (end == std::string::npos) return default_val;

    return json.substr(start, end - start);
}

static int32_t json_get_int(const std::string& json, const std::string& key, int32_t default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;

    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    size_t start = pos;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-')) pos++;

    if (start == pos) return default_val;

    try {
        return std::stoi(json.substr(start, pos - start));
    } catch (...) {
        return default_val;
    }
}

static float json_get_float(const std::string& json, const std::string& key, float default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;

    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    size_t start = pos;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-' || json[pos] == '.')) pos++;

    if (start == pos) return default_val;

    try {
        return std::stof(json.substr(start, pos - start));
    } catch (...) {
        return default_val;
    }
}

static bool load_settings(const std::string& config_path, profile_test_settings* settings) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        LOGE("Failed to open settings file: %s\n", config_path.c_str());
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    file.close();

    settings->model_path = json_get_string(json, "model_path", settings->model_path);
    settings->mmproj_path = json_get_string(json, "mmproj_path", settings->mmproj_path);
    settings->backend = json_get_string(json, "backend", settings->backend);
    settings->vit_device = json_get_string(json, "vit_device", settings->vit_device);
    settings->htp_sessions = json_get_int(json, "htp_sessions", settings->htp_sessions);
    settings->n_ctx = json_get_int(json, "n_ctx", settings->n_ctx);
    settings->n_batch = json_get_int(json, "n_batch", settings->n_batch);
    settings->n_predict = json_get_int(json, "n_predict", settings->n_predict);
    settings->image_min_tokens = json_get_int(json, "image_min_tokens", settings->image_min_tokens);
    settings->top_k = json_get_int(json, "top_k", settings->top_k);
    settings->top_p = json_get_float(json, "top_p", settings->top_p);
    settings->temperature = json_get_float(json, "temperature", settings->temperature);
    settings->repetition_penalty = json_get_float(json, "repetition_penalty", settings->repetition_penalty);
    settings->image_dir = json_get_string(json, "image_dir", settings->image_dir);
    settings->htp_profile = json_get_int(json, "htp_profile", settings->htp_profile);
    settings->system_prompt = json_get_string(json, "system_prompt", settings->system_prompt);
    settings->user_prompt = json_get_string(json, "user_prompt", settings->user_prompt);
    settings->output_csv = json_get_string(json, "output_csv", settings->output_csv);
    settings->output_log = json_get_string(json, "output_log", settings->output_log);

    return true;
}

// ============================================================================
// Image Loading
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

struct image_data {
    std::string filename;
    uint8_t* rgb_data;
    int width;
    int height;
};

static std::vector<std::string> get_image_files(const std::string& dir_path) {
    std::vector<std::string> files;

    DIR* dir = opendir(dir_path.c_str());
    if (!dir) {
        LOGE("Failed to open image directory: %s\n", dir_path.c_str());
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        // Check for .jpg, .jpeg, .png extensions
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        if (lower_name.size() >= 4) {
            std::string ext = lower_name.substr(lower_name.size() - 4);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                files.push_back(dir_path + name);
            }
        }
    }

    closedir(dir);

    // Sort files for consistent ordering
    std::sort(files.begin(), files.end());

    return files;
}

static image_data* load_image(const std::string& path) {
    int width, height, channels;

#ifdef __ANDROID__
    // On Android, use Java file I/O or raw file descriptor
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        LOGE("Failed to open image: %s\n", path.c_str());
        return nullptr;
    }
#else
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        LOGE("Failed to open image: %s\n", path.c_str());
        return nullptr;
    }
#endif

    // Read file into memory
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> file_data(size);
    fread(file_data.data(), 1, size, f);
    fclose(f);

    // Decode image
    uint8_t* rgb = stbi_load_from_memory(
        file_data.data(), file_data.size(),
        &width, &height, &channels, 3  // Force RGB
    );

    if (!rgb) {
        LOGE("Failed to decode image: %s (stbi error: %s)\n", path.c_str(), stbi_failure_reason());
        return nullptr;
    }

    image_data* img = new image_data();
    img->filename = path.substr(path.find_last_of("/\\") + 1);
    img->rgb_data = rgb;
    img->width = width;
    img->height = height;

    LOGI("Loaded image: %s (%dx%d, %d channels)\n", img->filename.c_str(), width, height, channels);

    return img;
}

static void free_image(image_data* img) {
    if (img) {
        if (img->rgb_data) {
            stbi_image_free(img->rgb_data);
        }
        delete img;
    }
}

// ============================================================================
// CSV Output
// ============================================================================

static void write_csv_header(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        LOGE("Failed to create CSV file: %s\n", path.c_str());
        return;
    }

    fprintf(f, "Image,Backend,Model,PromptTokens,CompletionTokens,TotalTime_ms,TokensPerSecond\n");
    fclose(f);
}

static void write_csv_row(const std::string& path, const std::string& image, const std::string& backend,
                          const std::string& model, int32_t prompt_tokens, int32_t completion_tokens,
                          double total_time_ms, double tokens_per_second) {
    FILE* f = fopen(path.c_str(), "a");
    if (!f) {
        LOGE("Failed to append to CSV: %s\n", path.c_str());
        return;
    }

    // Extract just the filename from model path
    std::string model_name = model;
    size_t pos = model.find_last_of("/\\");
    if (pos != std::string::npos) {
        model_name = model.substr(pos + 1);
    }

    fprintf(f, "%s,%s,%s,%d,%d,%.2f,%.2f\n",
            image.c_str(), backend.c_str(), model_name.c_str(),
            prompt_tokens, completion_tokens, total_time_ms, tokens_per_second);

    fclose(f);
}

// ============================================================================
// Android Environment Variables (from test.cpp pattern)
// ============================================================================

// ============================================================================
// Thermal / Clock Monitor
// ============================================================================

// Samples temperatures and clocks on a background thread while one image runs.
// Everything read here is world-readable on the Fold 7; missing files just report 0.
class thermal_monitor {
public:
    thermal_monitor() {
        const std::string thermal_root = "/sys/class/thermal";
        for (const std::string& name : list_dir(thermal_root, "thermal_zone")) {
            const std::string zone = thermal_root + "/" + name;
            const std::string type = read_word(zone + "/type");
            if (type.rfind("gpuss-", 0) == 0) gpu_zones.push_back(zone + "/temp");
            else if (type.rfind("nsp", 0) == 0) npu_zones.push_back(zone + "/temp");
            else if (type.rfind("cpu-", 0) == 0) cpu_zones.push_back(zone + "/temp");
        }
        const std::string cpufreq_root = "/sys/devices/system/cpu/cpufreq";
        for (const std::string& name : list_dir(cpufreq_root, "policy")) {
            cpu_policies[name] = cpufreq_root + "/" + name + "/scaling_cur_freq";
        }
    }

    ~thermal_monitor() {
        running = false;
        if (worker.joinable()) worker.join();
    }

    void start() {
        gpu_temp_max = npu_temp_max = cpu_temp_max = 0;
        gpu_mhz_min = gpu_mhz_max = gpu_busy_mhz_sum = gpu_busy_samples = 0;
        samples = 0;
        cpu_mhz_sum.clear();
        running = true;
        worker = std::thread([this] {
            while (running) {
                sample();
                for (int i = 0; i < SAMPLE_PERIOD_STEPS && running; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(SAMPLE_STEP_MS));
                }
            }
        });
    }

    void stop_and_log(size_t image_index) {
        running = false;
        if (worker.joinable()) worker.join();

        std::string cpu_clocks;
        for (const auto& entry : cpu_mhz_sum) {
            cpu_clocks += " " + entry.first + "_mhz_avg=" + std::to_string(entry.second / std::max(samples, 1L));
        }
        LOGI("Thermal: image=%zu samples=%ld gpu_temp_max=%.1f npu_temp_max=%.1f cpu_temp_max=%.1f "
             "gpu_mhz_min=%ld gpu_mhz_max=%ld gpu_busy_mhz_avg=%ld gpu_busy_samples=%ld%s",
             image_index, samples, gpu_temp_max / 1000.0, npu_temp_max / 1000.0, cpu_temp_max / 1000.0,
             gpu_mhz_min, gpu_mhz_max, gpu_busy_mhz_sum / std::max(gpu_busy_samples, 1L), gpu_busy_samples,
             cpu_clocks.c_str());
    }

private:
    static constexpr int SAMPLE_STEP_MS = 50;       // stop() waits at most one step
    static constexpr int SAMPLE_PERIOD_STEPS = 10;  // 500 ms between samples
    static constexpr long GPU_BUSY_PERCENT = 50;    // below this the clock is not the encoder's

    static std::vector<std::string> list_dir(const std::string& path, const std::string& prefix) {
        std::vector<std::string> names;
        DIR* dir = opendir(path.c_str());
        if (!dir) return names;
        while (struct dirent* entry = readdir(dir)) {
            if (strncmp(entry->d_name, prefix.c_str(), prefix.size()) == 0) {
                names.push_back(entry->d_name);
            }
        }
        closedir(dir);
        return names;
    }

    static std::string read_word(const std::string& path) {
        std::ifstream file(path);
        std::string word;
        file >> word;
        return word;
    }

    static long read_long(const std::string& path) {
        std::ifstream file(path);
        long value = 0;
        file >> value;
        return value;
    }

    static long max_of(const std::vector<std::string>& paths, long current) {
        for (const std::string& path : paths) {
            current = std::max(current, read_long(path));
        }
        return current;
    }

    void sample() {
        gpu_temp_max = max_of(gpu_zones, gpu_temp_max);
        npu_temp_max = max_of(npu_zones, npu_temp_max);
        cpu_temp_max = max_of(cpu_zones, cpu_temp_max);

        const long gpu_mhz = read_long("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq") / 1000000;
        gpu_mhz_min = samples == 0 ? gpu_mhz : std::min(gpu_mhz_min, gpu_mhz);
        gpu_mhz_max = std::max(gpu_mhz_max, gpu_mhz);
        if (read_long("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage") >= GPU_BUSY_PERCENT) {
            gpu_busy_mhz_sum += gpu_mhz;
            gpu_busy_samples++;
        }

        for (const auto& policy : cpu_policies) {
            cpu_mhz_sum[policy.first] += read_long(policy.second) / 1000;
        }
        samples++;
    }

    std::vector<std::string> gpu_zones, npu_zones, cpu_zones;
    std::map<std::string, std::string> cpu_policies;
    std::map<std::string, long> cpu_mhz_sum;
    long gpu_temp_max = 0, npu_temp_max = 0, cpu_temp_max = 0;
    long gpu_mhz_min = 0, gpu_mhz_max = 0, gpu_busy_mhz_sum = 0, gpu_busy_samples = 0;
    long samples = 0;
    std::atomic<bool> running{false};
    std::thread worker;
};

static void set_android_env_vars(const std::string& device, int htp_profile) {
    // LD_LIBRARY_PATH: current dir for finding .so files
    setenv("LD_LIBRARY_PATH", ".:/vendor/lib64", 1);
    // ADSP_LIBRARY_PATH: needed for Hexagon backend
    setenv("ADSP_LIBRARY_PATH", ".", 1);
    // The Hexagon session count is set by mtmd_inference_init from the device strings
    if (device.find("HTP") != std::string::npos) {
        // GGML_HEXAGON_PROFILE: enable HTP per-op profiling
        if (htp_profile > 0) {
            std::string profile_val = std::to_string(htp_profile);
            setenv("GGML_HEXAGON_PROFILE", profile_val.c_str(), 1);
            LOGI("Set GGML_HEXAGON_PROFILE=%d (from config)", htp_profile);
        }
    }
}

// ============================================================================
// Main Test Function
// ============================================================================

int main(int argc, char** argv) {
    LOGI("=== mtmd-profile-test START ===");

    // Default settings
    profile_test_settings settings;
    profile_test_settings_default(&settings);

    // Load settings from file first, so command line flags override them
    std::string config_path;
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "--config") {
            config_path = argv[i + 1];
        }
    }
    if (!config_path.empty()) {
        if (!load_settings(config_path, &settings)) {
            LOGE("Failed to load settings from: %s\n", config_path.c_str());
            return 1;
        }
    }

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            ++i;  // Loaded above
        } else if (arg == "--model" && i + 1 < argc) {
            settings.model_path = argv[++i];
        } else if (arg == "--mmproj" && i + 1 < argc) {
            settings.mmproj_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            settings.backend = argv[++i];
        } else if (arg == "--vit-device" && i + 1 < argc) {
            settings.vit_device = argv[++i];
        } else if (arg == "--image-dir" && i + 1 < argc) {
            settings.image_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            settings.output_csv = argv[++i];
        } else if (arg == "--htp-profile" && i + 1 < argc) {
            settings.htp_profile = std::stoi(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            // Handled in params
        }
    }

    // Log settings
    LOGI("Settings:");
    LOGI("  Model: %s", settings.model_path.c_str());
    LOGI("  MMProj: %s", settings.mmproj_path.c_str());
    LOGI("  Backend: %s", settings.backend.c_str());
    LOGI("  ViT device: %s", settings.vit_device.empty() ? "default" : settings.vit_device.c_str());
    LOGI("  HTP Sessions: %d", settings.htp_sessions);
    LOGI("  N Context: %d", settings.n_ctx);
    LOGI("  N Batch: %d", settings.n_batch);
    LOGI("  N Predict: %d", settings.n_predict);
    LOGI("  Image Dir: %s", settings.image_dir.c_str());
    LOGI("  Output CSV: %s", settings.output_csv.c_str());
    LOGI("Note: Set GGML_HEXAGON_PROFILE=1 in shell before running for HTP per-op profiling");

    // Get image files
    std::vector<std::string> image_files = get_image_files(settings.image_dir);
    if (image_files.empty()) {
        LOGE("No image files found in: %s\n", settings.image_dir.c_str());
        return 1;
    }

    LOGI("Found %zu image(s) to test", image_files.size());

    // Initialize inference params
    mtmd_inference_params_t params;
    mtmd_inference_params_default(&params);

    params.model.path = settings.model_path;
    params.model.mmproj_path = settings.mmproj_path;
    params.model.mmproj_device = settings.vit_device;
    params.n_ctx = settings.n_ctx;
    params.n_batch = settings.n_batch;
    params.n_predict = settings.n_predict;
    params.cpuparams.n_threads = 4;

    // Set backend device
    if (settings.backend == "CPU") {
        params.device = "CPU";
        params.n_gpu_layers = 0;
    } else if (settings.backend == "GPU") {
        params.device = "GPUOpenCL";
        params.n_gpu_layers = 999;
    } else if (settings.backend == "NPU-4Core") {
        params.device = "HTP0,HTP1,HTP2,HTP3";
        params.n_gpu_layers = 999;
    } else if (settings.backend == "NPU-1Core") {
        params.device = "HTP0";
        params.n_gpu_layers = 999;
    } else {
        LOGE("Unknown backend: %s\n", settings.backend.c_str());
        return 1;
    }

    // Sampling params
    params.sampling.top_k = settings.top_k;
    params.sampling.top_p = settings.top_p;
    params.sampling.temp = settings.temperature;

    // Enable verbose logging for profiling
    params.show_timings = true;
    params.no_perf = false;
    params.htp_profile = settings.htp_profile;  // Set HTP profiling parameter
    // Debug and verbose logs emit about 1,900 lines per token, so keep them to profiling runs
    params.debug_logs_enabled = settings.htp_profile > 0;
    params.info_logs_enabled = true;   // Enable info logs
    params.verbose_prompt = settings.htp_profile > 0;

    // Set Android environment variables BEFORE initialization (critical for HTP profiling)
    set_android_env_vars(params.device, settings.htp_profile);

    // Initialize inference
    mtmd_inference_context_t* ctx = nullptr;
    mtmd_api_status status = mtmd_inference_init(params, &ctx, nullptr, nullptr);
    if (status != MTMD_API_SUCCESS) {
        LOGE("Failed to initialize inference: %s\n", mtmd_api_status_string(status));
        return 1;
    }

    // Write CSV header
    write_csv_header(settings.output_csv);

    // Process each image
    thermal_monitor thermal;
    for (size_t i = 0; i < image_files.size(); i++) {
        image_data* img = load_image(image_files[i]);
        if (!img) {
            LOGE("Failed to load image: %s\n", image_files[i].c_str());
            continue;
        }

        LOGI("=== Processing image %zu/%zu: %s ===", i + 1, image_files.size(), img->filename.c_str());

        // Prepare image data
        mtmd_image_data_t image;
        image.width = img->width;
        image.height = img->height;
        image.data = img->rgb_data;

        // Prepare prompt
        mtmd_prompt_params_t prompt;
        prompt.system_prompt = settings.system_prompt;
        prompt.user_prompt = settings.user_prompt;

        // Run inference
        mtmd_inference_result_t result;
        thermal.start();
        status = mtmd_inference_run(ctx, &image, 1, prompt, &result, nullptr, nullptr);
        thermal.stop_and_log(i + 1);

        if (status != MTMD_API_SUCCESS) {
            LOGE("Inference failed for %s: %s\n", img->filename.c_str(), mtmd_api_status_string(status));
            free_image(img);
            continue;
        }

        // Log results
        LOGI("Results for %s:", img->filename.c_str());
        LOGI("  Prompt tokens: %d", result.n_prompt_tokens);
        LOGI("  Completion tokens: %d", result.n_tokens);
        LOGI("  Total time: %.2f ms", result.prompt_eval_time_ms + result.eval_time_ms);
        LOGI("  Tokens per second: %.2f", result.tokens_per_second);
        LOGI("  Response: %.200s...", result.text.c_str());

        // Write to CSV
        double total_time = result.prompt_eval_time_ms + result.eval_time_ms;
        write_csv_row(settings.output_csv, img->filename, settings.backend,
                      settings.model_path, result.n_prompt_tokens, result.n_tokens,
                      total_time, result.tokens_per_second);

        free_image(img);
    }

    // Cleanup
    mtmd_inference_free(ctx);

    LOGI("=== mtmd-profile-test COMPLETE ===");
    LOGI("Results written to: %s", settings.output_csv.c_str());

    return 0;
}
