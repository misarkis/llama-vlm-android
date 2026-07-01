// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// mtmd-inference-api.cpp
// Implementation of multimodal inference API
// Wraps mtmd-cli functionality into clean API calls
// Images are passed as bitmap data in memory, not file paths

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

#include <cstdio>
#include <cstring>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "mtmd-inference-api"

#define GGML_LOG_INFO(...)  do { if (g_mtmd_info_logs) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while(0)
#define GGML_LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define GGML_LOG_WARN(...)  __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) do { if (g_mtmd_debug_logs) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } while(0)
#else
#include <cstdio>
#define GGML_LOG_INFO(...)  printf(__VA_ARGS__)
#define GGML_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_WARN(...)  fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) printf(__VA_ARGS__)
#endif
#include <limits.h>
#include <cinttypes>
#include <ctime>
#include <chrono>

// Global log flags for Android logging (set during init from params)
#ifdef __ANDROID__
static bool g_mtmd_debug_logs = false;
static bool g_mtmd_info_logs = false;

static void android_log_callback(ggml_log_level level, const char* text, void* user_data) {
    android_LogPriority priority = ANDROID_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: priority = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  priority = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:
            if (!g_mtmd_info_logs) return;  // Skip info logs if disabled
            priority = ANDROID_LOG_INFO;
            break;
        case GGML_LOG_LEVEL_DEBUG:
		default:
            if (!g_mtmd_debug_logs) return;  // Skip debug logs if disabled
            priority = ANDROID_LOG_DEBUG;
            break;
    }
    __android_log_print(priority, LOG_TAG, "%s", text);
}
#endif


// Helper to check for antiprompt in generated tokens (mimic mtmd_cli_context::check_antiprompt)
static bool check_antiprompt(const llama_tokens& generated_tokens, const llama_tokens& antiprompt_tokens) {
    if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
        return false;
    }
    return std::equal(
        generated_tokens.end() - antiprompt_tokens.size(),
        generated_tokens.end(),
        antiprompt_tokens.begin()
    );
}

// Internal context structure
struct mtmd_inference_context {
    // From mtmd_cli_context
    mtmd::context_ptr ctx_vision;
    common_init_result_ptr llama_init;

    llama_model* model;
    llama_context* lctx;
    const llama_vocab* vocab;
    common_sampler* smpl;
    llama_batch batch;
    int n_batch;

    mtmd::bitmaps bitmaps;

    // Chat templates
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja;

    // Support for legacy templates
    llama_tokens antiprompt_tokens;

    // Debug logging flag (copy from params for quick access)
    bool verbose_prompt;
    bool debug_logs_enabled;
    bool info_logs_enabled;

    int n_threads;
    llama_pos n_past;

    common_debug_cb_user_data cb_data;

    // Parameters (for reference)
    mtmd_inference_params_t params;

    // Callbacks
    mtmd_progress_callback_t progress_cb;
    void* progress_user_data;
    mtmd_token_callback_t token_cb;
    void* token_user_data;

    mtmd_inference_context()
        : model(nullptr)
        , lctx(nullptr)
        , vocab(nullptr)
        , smpl(nullptr)
        , n_batch(512)
        , use_jinja(false)
        , verbose_prompt(false)
        , debug_logs_enabled(false)
        , info_logs_enabled(false)
		, n_threads(4)
		, n_past(0)
    {}
};

// ============================================================================
// Default Parameters
// ============================================================================

void mtmd_cpu_params_default(mtmd_cpu_params_t* params) {
    params->n_threads = 4;
    params->priority = 0;
    params->noprint = false;
    params->no_perf = false;
}

void mtmd_sampling_params_default(mtmd_sampling_params_t* params) {
    params->seed = LLAMA_DEFAULT_SEED;
    params->n_prev = 64;
    params->n_probs = 0;
    params->min_keep = 0;
    params->top_k = 40;
    params->top_p = 0.95f;
    params->min_p = 0.05f;
    params->xtc_probability = 0.00f;
    params->xtc_threshold = 0.10f;
    params->typ_p = 1.00f;
    params->temp = 0.80f;
    params->dynatemp_range = 0.00f;
    params->dynatemp_exponent = 1.00f;
    params->penalty_last_n = 64;
    params->penalty_repeat = 1.00f;
    params->penalty_freq = 0.00f;
    params->penalty_present = 0.00f;
    params->dry_multiplier = 0.0f;
    params->dry_base = 1.75f;
    params->dry_allowed_length = 2;
    params->dry_penalty_last_n = -1;
    params->dry_sequence_breakers = {"\n", ":", "\"", "*"}; // DRY sequence breakers
    params->adaptive_target = -1.0f;
    params->adaptive_decay = 0.90f;
    params->mirostat = 0;
    params->top_n_sigma = -1.00f;
    params->mirostat_tau = 5.00f;
    params->mirostat_eta = 0.10f;
    params->ignore_eos = false;
    params->no_perf = false;
    params->timing_per_token = false;
    params->grammar = "";
    params->grammar_lazy = false;
    params->grammar_triggers.clear();
    params->preserved_tokens.clear();
    params->logit_bias.clear();
    params->logit_bias_eog.clear();
    params->generation_prompt = "";
    params->backend_sampling = false;

    // Reasoning budget
    params->reasoning_budget_tokens = -1;  // -1 = disabled
    params->reasoning_budget_message = ""; // message injected when budget exhausted
    params->reasoning_budget_start.clear();
    params->reasoning_budget_end.clear();
    params->reasoning_budget_forced.clear();
}

void mtmd_inference_params_default(mtmd_inference_params_t* params) {
    // Model
    params->model.path = "";
    params->model.mmproj_path = "";
    params->model.mmproj_use_gpu = true;
    params->model.no_mmproj = false;

    // Backend
    params->device = "CPU";
    params->library_directory = "";  // Empty = use system default
    params->n_gpu_layers = 999;  // Offload all layers by default

    // Context
    params->n_ctx = 16384;
    params->n_batch = 512;
    params->n_ubatch = 512;

    // Generation
    params->n_predict = -1;  // Unlimited
    params->n_keep = 0;
    params->n_draft = 0;     // No speculative decoding by default

    // CPU
    mtmd_cpu_params_default(&params->cpuparams);
    mtmd_cpu_params_default(&params->cpuparams_batch);

    // Sampling
    mtmd_sampling_params_default(&params->sampling);

    // Speculative
    params->speculative.n_max = 3;
    params->speculative.n_min = 0;
    params->speculative.p_split = 0.1f;
    params->speculative.p_min = 0.0f;
    params->speculative.draft_model_path = "";
    params->speculative.n_gpu_layers_draft = -1;

    // Multimodal
    params->image_min_tokens = -1;
    params->image_max_tokens = -1;
    params->no_mmproj_offload = false;

    // Performance/logging
    params->special = false;
    params->verbose_prompt = false;
    params->display_prompt = false;
    params->no_perf = false;
    params->show_timings = true;
    params->warmup = false;

    // Memory
    params->use_mmap = false;
    params->use_mlock = false;
    params->no_kv_offload = false;
    params->swa_full = false;
    params->kv_unified = false;

    // Flash attention (0 = disabled, 1 = enabled, 2 = auto)
    params->flash_attn_type = 1;  // Auto (llama.cpp default)

    // Chat template
    params->chat_template = "";
    params->use_jinja = false;

    // Chat history persistence (default: false = clear after each inference)
    params->keep_chat_history = false;

    // Backend init
    params->backend_init = true;
    params->print_system_info = true;

    // Debug logging (default: disabled)
    params->debug_logs_enabled = false;

    // Info logging (default: disabled)
    params->info_logs_enabled = false;
}

// ============================================================================
// Helper Functions
// ============================================================================

static void set_env_var(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<double>(ms.count());
}

// Parse comma-separated device names into device pointers
// Matches llama.cpp's parse_device_list behavior
static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    if (value.empty()) {
        return devices;  // Empty = auto-detect (default llama.cpp behavior)
    }

    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);  // Disable offloading
        return devices;
    }

    for (const auto & dev_name : dev_names) {
        std::string trimmed = string_strip(dev_name);
        if (trimmed.empty()) continue;

        auto * dev = ggml_backend_dev_by_name(trimmed.c_str());
        if (!dev) {
            GGML_LOG_WARN("Warning: device '%s' not found, skipping", trimmed.c_str());
            continue;
        }
        devices.push_back(dev);
        GGML_LOG_INFO("Device '%s' found: %s", trimmed.c_str(), ggml_backend_dev_description(dev));
    }

    // Add nullptr at end to signal CPU for remaining layers (llama.cpp convention)
    if (!devices.empty()) {
        devices.push_back(nullptr);
    }

    return devices;
}

// ============================================================================
// API Implementation
// ============================================================================

const char* mtmd_api_status_string(mtmd_api_status status) {
    switch (status) {
        case MTMD_API_SUCCESS: return "Success";
        case MTMD_API_ERROR_GENERIC: return "Generic error";
        case MTMD_API_ERROR_INVALID_PARAM: return "Invalid parameter";
        case MTMD_API_ERROR_MODEL_LOAD: return "Failed to load model";
        case MTMD_API_ERROR_BACKEND_INIT: return "Failed to initialize backend";
        case MTMD_API_ERROR_IMAGE_LOAD: return "Failed to load image";
        case MTMD_API_ERROR_INFERENCE: return "Inference error";
        case MTMD_API_ERROR_NOT_INITIALIZED: return "Context not initialized";
        case MTMD_API_ERROR_ALREADY_INITIALIZED: return "Context already initialized";
        default: return "Unknown error";
    }
}

mtmd_api_status mtmd_inference_init(
    const mtmd_inference_params_t& params,
    mtmd_inference_context_t** ctx_out,
    mtmd_progress_callback_t progress_cb,
    void* user_data)
{
    GGML_LOG_INFO("=== mtmd_inference_init START ===");

#ifdef __ANDROID__
    // Route llama.cpp logs to Android logcat
    llama_log_set(android_log_callback, nullptr);

    // Route mtmd/clip logs to Android logcat (for vision encoder backend info)
    mtmd_log_set(android_log_callback, nullptr);

    GGML_LOG_INFO("Android log callback installed (llama + mtmd)");
#endif

    double init_start = get_time_ms();

    if (ctx_out == nullptr) {
        GGML_LOG_ERROR("Error: ctx_out is null");
        return MTMD_API_ERROR_INVALID_PARAM;
    }

    if (params.model.path.empty() || params.model.mmproj_path.empty()) {
        GGML_LOG_ERROR("Error: Model path and mmproj path are required");
        return MTMD_API_ERROR_INVALID_PARAM;
    }

    GGML_LOG_INFO("Model: %s", params.model.path.c_str());
    GGML_LOG_INFO("MMProj: %s", params.model.mmproj_path.c_str());
    GGML_LOG_INFO("Device: %s, GPU layers: %d", params.device.c_str(), params.n_gpu_layers);
    GGML_LOG_INFO("Context: %d, Batch: %d, Predict: %d", params.n_ctx, params.n_batch, params.n_predict);
    GGML_LOG_INFO("Verbose: %d", params.verbose_prompt);

    // Create context
    mtmd_inference_context_t* ctx = new mtmd_inference_context();
    ctx->params = params;
    ctx->progress_cb = progress_cb;
    ctx->progress_user_data = user_data;

    if (progress_cb) {
        progress_cb("Initializing inference engine...", user_data);
    }

    // Set environment variables and load backends
    if (progress_cb) {
        progress_cb("Setting environment variables...", user_data);
    }

    // Set device
    std::string device = params.device.empty() ? "CPU" : params.device;
	
	// Set D based on device
	setenv("D", params.device.c_str(),1);
    GGML_LOG_INFO("Set D=%s", params.device.c_str());

    // Set GGML_HEXAGON_NDEV based on device
    if (params.device.find("HTP0,HTP1,HTP2,HTP3") != std::string::npos) {
        setenv("GGML_HEXAGON_NDEV", "4",1);
        GGML_LOG_INFO("Set GGML_HEXAGON_NDEV=4 for NPU backend");
    }else if (params.device.find("HTP0,HTP1,HTP2") != std::string::npos) {
        setenv("GGML_HEXAGON_NDEV", "3",1);
        GGML_LOG_INFO("Set GGML_HEXAGON_NDEV=3 for NPU backend");
    }else if (params.device.find("HTP0,HTP1") != std::string::npos) {
        setenv("GGML_HEXAGON_NDEV", "2",1);
        GGML_LOG_INFO("Set GGML_HEXAGON_NDEV=2 for NPU backend");
    }else if (params.device.find("HTP0") != std::string::npos) {
        setenv("GGML_HEXAGON_NDEV", "1",1);
        GGML_LOG_INFO("Set GGML_HEXAGON_NDEV=1 for NPU backend");
    }else {
        setenv("GGML_HEXAGON_NDEV", "0",1);
        GGML_LOG_INFO("Set GGML_HEXAGON_NDEV=0 for CPU/GPU backend");
    }

    if(params.verbose_prompt){
		setenv("GGML_HEXAGON_VERBOSE", "1",1);
	}

    if (progress_cb) {
        progress_cb("Initializing backends...", user_data);
    }

     // Build common_params from our parameters
    common_params cparams;

    // Model paths
    cparams.model.path = params.model.path;
    cparams.mmproj.path = params.model.mmproj_path;
    cparams.mmproj_use_gpu = params.model.mmproj_use_gpu;
    cparams.no_mmproj = params.model.no_mmproj;

    // Context
    cparams.n_ctx = params.n_ctx;
    cparams.n_batch = params.n_batch;
    cparams.n_ubatch = params.n_ubatch > 0 ? params.n_ubatch : params.n_batch;

    // GPU layers
    cparams.n_gpu_layers = params.n_gpu_layers;

    // Initialize backend if requested
    if (params.backend_init) {
        // Load backends from library_directory if provided
        if (!params.library_directory.empty()) {
            GGML_LOG_INFO("Loading backends from: %s", params.library_directory.c_str());
            ggml_backend_load_all_from_path(params.library_directory.c_str());
        }

        GGML_LOG_INFO("Backend reg count: %zu", ggml_backend_reg_count());

        // llama_backend_init is idempotent - safe to call multiple times
        llama_backend_init();
        GGML_LOG_INFO("llama_backend_init completed, reg count: %zu", ggml_backend_reg_count());
    }

    
	// Device selection - parse device string and set devices
    if (!params.device.empty() && params.device != "CPU") {
        cparams.devices = parse_device_list(params.device);
        if (!cparams.devices.empty()) {
            GGML_LOG_INFO("Devices set: %zu device(s)", cparams.devices.size());
        }
    } else if (params.device == "CPU" || params.device.empty()) {
        // CPU mode - disable GPU offloading
        cparams.n_gpu_layers = 0;
        GGML_LOG_INFO("CPU mode: GPU offloading disabled");
    }

    // CPU params
    cparams.cpuparams.n_threads = params.cpuparams.n_threads;
    cparams.cpuparams.priority = (ggml_sched_priority) params.cpuparams.priority;

    cparams.cpuparams_batch.n_threads = params.cpuparams_batch.n_threads;

    // Sampling params
    cparams.sampling.seed = params.sampling.seed;
    cparams.sampling.n_prev = params.sampling.n_prev;
    cparams.sampling.n_probs = params.sampling.n_probs;
    cparams.sampling.min_keep = params.sampling.min_keep;
    cparams.sampling.top_k = params.sampling.top_k;
    cparams.sampling.top_p = params.sampling.top_p;
    cparams.sampling.min_p = params.sampling.min_p;
    cparams.sampling.xtc_probability = params.sampling.xtc_probability;
    cparams.sampling.xtc_threshold = params.sampling.xtc_threshold;
    cparams.sampling.typ_p = params.sampling.typ_p;
    cparams.sampling.temp = params.sampling.temp;
    cparams.sampling.dynatemp_range = params.sampling.dynatemp_range;
    cparams.sampling.dynatemp_exponent = params.sampling.dynatemp_exponent;
    cparams.sampling.penalty_last_n = params.sampling.penalty_last_n;
    cparams.sampling.penalty_repeat = params.sampling.penalty_repeat;
    cparams.sampling.penalty_freq = params.sampling.penalty_freq;
    cparams.sampling.penalty_present = params.sampling.penalty_present;
    cparams.sampling.dry_multiplier = params.sampling.dry_multiplier;
    cparams.sampling.dry_base = params.sampling.dry_base;
    cparams.sampling.dry_allowed_length = params.sampling.dry_allowed_length;
    cparams.sampling.dry_penalty_last_n = params.sampling.dry_penalty_last_n;
    cparams.sampling.dry_sequence_breakers = params.sampling.dry_sequence_breakers;
    cparams.sampling.adaptive_target = params.sampling.adaptive_target;
    cparams.sampling.adaptive_decay = params.sampling.adaptive_decay;
    cparams.sampling.mirostat = params.sampling.mirostat;
    cparams.sampling.top_n_sigma = params.sampling.top_n_sigma;
    cparams.sampling.mirostat_tau = params.sampling.mirostat_tau;
    cparams.sampling.mirostat_eta = params.sampling.mirostat_eta;
    cparams.sampling.ignore_eos = params.sampling.ignore_eos;
    cparams.sampling.no_perf = params.sampling.no_perf;
    cparams.sampling.timing_per_token = params.sampling.timing_per_token;
    cparams.sampling.grammar_lazy = params.sampling.grammar_lazy;
    cparams.sampling.backend_sampling = params.sampling.backend_sampling;
    cparams.sampling.reasoning_budget_tokens = params.sampling.reasoning_budget_tokens;
    cparams.sampling.reasoning_budget_message = params.sampling.reasoning_budget_message;

    // Grammar
    if (!params.sampling.grammar.empty()) {
        cparams.sampling.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, params.sampling.grammar);
    }

    // Logit bias
    cparams.sampling.logit_bias = params.sampling.logit_bias;

    // Generation settings
    cparams.n_predict = params.n_predict;
    cparams.n_keep = params.n_keep;

    // Multimodal
    cparams.image_min_tokens = params.image_min_tokens;
    cparams.image_max_tokens = params.image_max_tokens;

    // Performance/logging
    cparams.use_color = params.use_color;
    cparams.special = params.special;
    cparams.verbose_prompt = params.verbose_prompt;
    cparams.display_prompt = params.display_prompt;
    cparams.no_perf = params.no_perf;
    cparams.show_timings = params.show_timings;
    cparams.warmup = params.warmup;

    // Log batch configuration for performance analysis
    GGML_LOG_INFO("  n_batch: %d", cparams.n_batch);
    GGML_LOG_INFO("  n_ubatch: %d", cparams.n_ubatch);
    GGML_LOG_INFO("  n_ctx: %d", cparams.n_ctx);

    // Memory
    cparams.use_mmap = params.use_mmap;
    cparams.use_mlock = params.use_mlock;
    cparams.no_kv_offload = params.no_kv_offload;
    cparams.swa_full = params.swa_full;
    cparams.kv_unified = params.kv_unified;

    // Flash attention
    cparams.flash_attn_type = (enum llama_flash_attn_type) params.flash_attn_type;

    // Log flash attention configuration
    GGML_LOG_INFO("  Flash attention type: %d", params.flash_attn_type);
    GGML_LOG_INFO("  Flash attention enabled: %s", params.flash_attn_type > 0 ? "yes" : "no");

	// Set global debug flag for Android logging callback
#ifdef __ANDROID__
    g_mtmd_debug_logs = params.debug_logs_enabled;
    g_mtmd_info_logs = params.info_logs_enabled;
#endif

    // Chat template
    cparams.chat_template = params.chat_template;
    cparams.use_jinja = params.use_jinja;

    if (progress_cb) {
        progress_cb("Loading model...", user_data);
    }

    // Log model paths for debugging
    GGML_LOG_INFO("Model path: %s", params.model.path.c_str());
    GGML_LOG_INFO("MMProj path: %s", params.model.mmproj_path.c_str());
    GGML_LOG_INFO("Device: %s, GPU layers: %d", params.device.c_str(), params.n_gpu_layers);
    

    // Initialize llama context from params
    ctx->llama_init = common_init_from_params(cparams);
    

    if (!ctx->llama_init) {
        GGML_LOG_ERROR("common_init_from_params returned null (init result)");
        
        delete ctx;
        return MTMD_API_ERROR_MODEL_LOAD;
    }

    if (!ctx->llama_init->model()) {
        GGML_LOG_ERROR("common_init_from_params returned null model");
        
        delete ctx;
        return MTMD_API_ERROR_MODEL_LOAD;
    }

    if (!ctx->llama_init->context()) {
        GGML_LOG_ERROR("common_init_from_params returned null context");
        
        delete ctx;
        return MTMD_API_ERROR_MODEL_LOAD;
    }

    GGML_LOG_INFO("Model loaded successfully");

    GGML_LOG_INFO("Initializing sampler...");

    ctx->model = ctx->llama_init->model();
    ctx->lctx = ctx->llama_init->context();
    ctx->vocab = llama_model_get_vocab(ctx->model);

    if (progress_cb) {
        progress_cb("Initializing sampler...", user_data);
    }

    // Initialize sampler
    ctx->smpl = common_sampler_init(ctx->model, cparams.sampling);
    if (!ctx->smpl) {
        GGML_LOG_ERROR("Failed to initialize sampler");
        llama_backend_free();
        delete ctx;
        return MTMD_API_ERROR_BACKEND_INIT;
    }

    // Initialize mtmd context
    if (progress_cb) {
        progress_cb("Initializing multimodal context...", user_data);
    }

    GGML_LOG_INFO("Initializing multimodal context...");

    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.model.mmproj_use_gpu && params.n_gpu_layers > 0;
    mtmd_params.print_timings = params.show_timings;
    mtmd_params.n_threads = params.cpuparams.n_threads;
    mtmd_params.warmup = params.warmup;

    ctx->ctx_vision.reset(mtmd_init_from_file(
        params.model.mmproj_path.c_str(),
        ctx->model,
        mtmd_params
    ));

    if (!ctx->ctx_vision) {
        GGML_LOG_ERROR("Failed to initialize mtmd context");
        common_sampler_free(ctx->smpl);
        llama_backend_free();
        delete ctx;
        return MTMD_API_ERROR_BACKEND_INIT;
    }

    // Initialize chat history
    ctx->chat_history.clear();
    ctx->use_jinja = params.use_jinja;
    ctx->verbose_prompt = params.verbose_prompt;
    ctx->debug_logs_enabled = params.debug_logs_enabled;
    ctx->info_logs_enabled = params.info_logs_enabled;

    // Initialize chat templates (FIX: was missing, causing crash)
    GGML_LOG_INFO("Initializing chat templates...");
    ctx->tmpls = common_chat_templates_init(ctx->model, params.chat_template);
    if (!ctx->tmpls) {
        GGML_LOG_ERROR("Failed to initialize chat templates");
        GGML_LOG_ERROR("Model: %s", params.model.path.c_str());
        // Check if model has embedded template
        const char* embedded_template = llama_model_chat_template(ctx->model, nullptr);
        if (!embedded_template) {
            GGML_LOG_ERROR("Model does not have embedded chat template");
            GGML_LOG_ERROR("  For old llava models, use 'vicuna'");
            GGML_LOG_ERROR("  For MobileVLM models, use 'deepseek'");
        } else {
            GGML_LOG_ERROR("Model has embedded template, but initialization failed");
        }
        common_sampler_free(ctx->smpl);
        llama_backend_free();
        delete ctx;
        return MTMD_API_ERROR_MODEL_LOAD;
    }
    GGML_LOG_INFO("Chat template initialized, use_jinja=%d", ctx->use_jinja);

    // Load antiprompt tokens for legacy templates (mimic mtmd-cli)
    // Note: Modern templates (Jinja-based) rely on EOG tokens, antiprompt is fallback only
    if (!params.chat_template.empty()) {
        std::string template_name = params.chat_template;

        // Normalize template name for comparison
        if (template_name == "vicuna" || template_name.find("vicuna") != std::string::npos) {
            ctx->antiprompt_tokens = common_tokenize(ctx->lctx, "ASSISTANT:", false, true);
            GGML_LOG_INFO("Set antiprompt for vicuna template");
        } else if (template_name == "deepseek" || template_name.find("deepseek") != std::string::npos) {
            ctx->antiprompt_tokens = common_tokenize(ctx->lctx, "###", false, true);
            GGML_LOG_INFO("Set antiprompt for deepseek template");
        }
        // Add more legacy templates as needed
    }

    // Initialize batch
    ctx->n_batch = params.n_batch;
    ctx->batch = llama_batch_init(1, 0, 1);
    ctx->n_past = 0;
    ctx->n_threads = params.cpuparams.n_threads;

    double init_end = get_time_ms();

    if (progress_cb) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Initialization complete in %.0f ms", init_end - init_start);
        progress_cb(msg, user_data);
    }

    *ctx_out = ctx;
    return MTMD_API_SUCCESS;
}

mtmd_api_status mtmd_inference_run(
    mtmd_inference_context_t* ctx,
    const mtmd_image_data_t* images,
    int32_t n_images,
    const mtmd_prompt_params_t& prompt,
    mtmd_inference_result_t* result_out,
    mtmd_token_callback_t token_cb,
    void* user_data)
{
    GGML_LOG_INFO("=== mtmd_inference_run START === images=%d, prompt=%zu chars, system=%zu chars",
            n_images, prompt.user_prompt.length(), prompt.system_prompt.length());

    if (ctx == nullptr || result_out == nullptr) {
        GGML_LOG_ERROR("ctx or result_out is null");
        return MTMD_API_ERROR_INVALID_PARAM;
    }

    if (ctx->model == nullptr || ctx->lctx == nullptr || ctx->ctx_vision == nullptr) {
        GGML_LOG_ERROR("Context not initialized");
        return MTMD_API_ERROR_NOT_INITIALIZED;
    }

    if (images == nullptr || n_images <= 0) {
        GGML_LOG_ERROR("No images provided");
        return MTMD_API_ERROR_INVALID_PARAM;
    }

    ctx->token_cb = token_cb;
    ctx->token_user_data = user_data;

    result_out->text = "";
    result_out->n_tokens = 0;
    result_out->load_time_ms = 0;
    result_out->prompt_eval_time_ms = 0;
    result_out->eval_time_ms = 0;
    result_out->tokens_per_second = 0.0;

    // Clear chat history if not keeping it (default behavior)
    if (!ctx->params.keep_chat_history) {
        ctx->chat_history.clear();
    }

    // Add system message to chat history if provided (mimic mtmd-cli eval_system_prompt_if_present)
    if (!prompt.system_prompt.empty()) {
        common_chat_msg system_msg;
        system_msg.role = "system";
        system_msg.content = prompt.system_prompt;
        ctx->chat_history.push_back(std::move(system_msg));
    }

    // Build user content with media marker if needed
    const char* media_marker = mtmd_default_marker();
    std::string user_content = prompt.user_prompt;

    bool has_media_marker = user_content.find(media_marker) != std::string::npos;
    if (!has_media_marker && n_images > 0) {
        // Add media marker at end of user content
        user_content += " " + std::string(media_marker);
    }

    // Create user message
    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = user_content;

    // Track if this is first inference for BOS token (before adding to chat_history)
    bool is_first_inference = ctx->chat_history.empty();

    // Format using the model's chat template (mimic chat_add_and_format)
    // add_user_prefix = true means this is the first user message being formatted
    std::string formatted_prompt = common_chat_format_single(
        ctx->tmpls.get(),
        ctx->chat_history,
        user_msg,
        true,  // add_user_prefix - always true for single-turn inference
        ctx->use_jinja
    );

    // Add user message to chat history after formatting
    ctx->chat_history.push_back(std::move(user_msg));

    // Load images from memory as bitmaps
    for (int i = 0; i < n_images; i++) {
        const mtmd_image_data_t& img = images[i];
        if (img.data == nullptr || img.width == 0 || img.height == 0) {
            GGML_LOG_ERROR("Invalid image data at index %d", i);
            return MTMD_API_ERROR_IMAGE_LOAD;
        }

        // Create bitmap from RGB data in memory
        struct mtmd_bitmap* bitmap = mtmd_bitmap_init(img.width, img.height, img.data);
        if (!bitmap) {
            GGML_LOG_ERROR("Failed to create bitmap from image data at index %d", i);
            return MTMD_API_ERROR_IMAGE_LOAD;
        }
        ctx->bitmaps.entries.push_back(mtmd::bitmap(std::move(bitmap)));
    }

    // Evaluate prompt
    double eval_start = get_time_ms();

    // Use formatted prompt from chat template
    mtmd_input_text text;
    text.text = formatted_prompt.c_str();
    // Only add BOS token on first message (matches mtmd-cli: "bool add_bos = ctx.chat_history.empty();")
    text.add_special = is_first_inference;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx->bitmaps.c_ptr();

    int32_t res = mtmd_tokenize(
        ctx->ctx_vision.get(),
        chunks.ptr.get(),
        &text,
        bitmaps_c_ptr.data(),
        bitmaps_c_ptr.size()
    );

    if (res != 0) {
        GGML_LOG_ERROR("Tokenization failed (res = %d)", res);
        return MTMD_API_ERROR_INFERENCE;
    }

    ctx->bitmaps.entries.clear();

    // Reset n_past and KV cache for single-turn inference (keep_chat_history = false)
    if (!ctx->params.keep_chat_history) {
        ctx->n_past = 0;
        llama_memory_clear(llama_get_memory(ctx->lctx), true);
    }

    llama_pos old_n_past = ctx->n_past;
    llama_pos new_n_past;
    res = mtmd_helper_eval_chunks(
        ctx->ctx_vision.get(),
        ctx->lctx,
        chunks.ptr.get(),
        ctx->n_past,
        0,  // seq_id
        ctx->n_batch,
        true,  // logits_last
        &new_n_past
    );

    if (res != 0) {
        GGML_LOG_ERROR("Evaluation failed (res = %d)", res);
        return MTMD_API_ERROR_INFERENCE;
    }

    ctx->n_past = new_n_past;

    double eval_end = get_time_ms();
    result_out->prompt_eval_time_ms = eval_end - eval_start;
    result_out->n_prompt_tokens = new_n_past - old_n_past;  // Actual prompt tokens for this call

    // Calculate and log total prompt tokens (text + image)
    int32_t total_prompt_tokens = new_n_past - old_n_past;
    GGML_LOG_INFO("  Total prompt tokens (text + image): %d", total_prompt_tokens);

    // Generate response
    double gen_start = get_time_ms();
    int n_predict = ctx->params.n_predict < 0 ? INT_MAX : ctx->params.n_predict;

    // Track generated tokens for antiprompt check (mimic mtmd-cli)
    llama_tokens generated_tokens;

    // Generation loop (mimic mtmd-cli generate_response)
    for (int i = 0; i < n_predict; i++) {
        // Sample token
        llama_token token_id = common_sampler_sample(ctx->smpl, ctx->lctx, -1);

        // Accept the token
        common_sampler_accept(ctx->smpl, token_id, true);

        // Track generated token
        generated_tokens.push_back(token_id);

        // Check for EOG token (mimic mtmd-cli line 190)
        if (llama_vocab_is_eog(ctx->vocab, token_id)) {
            break;
        }

        // Fallback: check for antiprompt (legacy template support)
        if (check_antiprompt(generated_tokens, ctx->antiprompt_tokens)) {
            break;
        }

        // Debug: log token and EOG status (only when verbose_prompt is enabled)
        if (ctx->verbose_prompt) {
            std::string token_str = common_token_to_piece(ctx->lctx, token_id);
            bool is_eog = llama_vocab_is_eog(ctx->vocab, token_id);
            GGML_LOG_DEBUG("token_id=%d, is_eog=%d, token='%s'", token_id, is_eog, token_str.c_str());
        }

        // Convert token to string and add to result
        std::string token_str = common_token_to_piece(ctx->lctx, token_id);
        result_out->text += token_str;
        result_out->n_tokens++;

        // Call token callback if provided
        if (token_cb) {
            token_cb(token_str.c_str(), user_data);
        }

        // Eval token (mimic mtmd-cli)
        common_batch_clear(ctx->batch);
        common_batch_add(ctx->batch, token_id, ctx->n_past++, {0}, true);

        if (llama_decode(ctx->lctx, ctx->batch)) {
            GGML_LOG_ERROR("Decoding failed at token %d", i);
            break;
        }
    }

    double gen_end = get_time_ms();
    result_out->eval_time_ms = gen_end - gen_start;

    // Calculate throughput
    if (result_out->eval_time_ms > 0) {
        result_out->tokens_per_second = (result_out->n_tokens * 1000.0) / result_out->eval_time_ms;
    }

    // Calculate prefill throughput (text tokens only)
    double prefill_throughput = 0;
    if (result_out->n_prompt_tokens > 0 && result_out->prompt_eval_time_ms > 0) {
        prefill_throughput = (result_out->n_prompt_tokens * 1000.0) / result_out->prompt_eval_time_ms;
    }

    // Calculate decode throughput
    double decode_throughput = 0;
    if (result_out->eval_time_ms > 0) {
        decode_throughput = (result_out->n_tokens * 1000.0) / result_out->eval_time_ms;
    }

    GGML_LOG_INFO("=== mtmd_inference_run COMPLETE ===");
    GGML_LOG_INFO("  Prompt tokens: %d", result_out->n_prompt_tokens);
    GGML_LOG_INFO("  Completion tokens: %d", result_out->n_tokens);
    GGML_LOG_INFO("  Prefill time: %.2f ms", result_out->prompt_eval_time_ms);
    GGML_LOG_INFO("  Decode time: %.2f ms", result_out->eval_time_ms);
    GGML_LOG_INFO("  Total time: %.2f ms", result_out->prompt_eval_time_ms + result_out->eval_time_ms);
    GGML_LOG_INFO("  Prefill throughput: %.2f t/s", prefill_throughput);
    GGML_LOG_INFO("  Decode throughput: %.2f t/s", decode_throughput);
    GGML_LOG_INFO("  Overall throughput: %.2f t/s", result_out->tokens_per_second);

    return MTMD_API_SUCCESS;
}

void mtmd_inference_free(mtmd_inference_context_t* ctx) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->smpl) {
        common_sampler_free(ctx->smpl);
    }

    if (ctx->batch.token) {
        llama_batch_free(ctx->batch);
    }

    // ctx_vision, llama_init are smart pointers, will be freed automatically

    delete ctx;

    // Free backend (only once, at the end)
    llama_backend_free();
}
