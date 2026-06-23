// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// mtmd-inference-api.h
// Multimodal Inference API - Wrapper for mtmd-cli functionality
// Provides a clean API for initializing, running, and cleaning up multimodal inference
// without console interaction
//
// Images are passed as bitmap data in memory (RGB format), not file paths.
// Prompts are passed as system and user strings in memory.

#ifndef MTMD_INFERENCE_API_H
#define MTMD_INFERENCE_API_H

#include <stdint.h>
#include <stdbool.h>
#include <string>
#include <vector>

// Include llama.h for proper type definitions
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Error Codes
// ============================================================================
typedef enum {
    MTMD_API_SUCCESS = 0,
    MTMD_API_ERROR_GENERIC = -1,
    MTMD_API_ERROR_INVALID_PARAM = -2,
    MTMD_API_ERROR_MODEL_LOAD = -3,
    MTMD_API_ERROR_BACKEND_INIT = -4,
    MTMD_API_ERROR_IMAGE_LOAD = -5,
    MTMD_API_ERROR_INFERENCE = -6,
    MTMD_API_ERROR_NOT_INITIALIZED = -7,
    MTMD_API_ERROR_ALREADY_INITIALIZED = -8
} mtmd_api_status;

// ============================================================================
// Forward Declarations
// ============================================================================
struct mtmd_inference_context;
typedef struct mtmd_inference_context mtmd_inference_context_t;

// ============================================================================
// CPU Parameters (matching common_cpu_params)
// ============================================================================
typedef struct {
    int32_t n_threads;           // number of threads (-1 = auto)
    int32_t priority;            // process priority (0 = default)
    bool    noprint;             // disable printing
    bool    no_perf;             // disable performance metrics
} mtmd_cpu_params_t;

// ============================================================================
// Sampling Parameters (matching common_params_sampling)
// ============================================================================
typedef struct {
    uint32_t seed;                  // random seed (LLAMA_DEFAULT_SEED = 0xFFFFFFFF)

    int32_t n_prev;                 // number of previous tokens to remember
    int32_t n_probs;                // output probabilities of top n_probs tokens
    int32_t min_keep;               // minimum tokens to keep from samplers
    int32_t top_k;                  // top-k sampling (<= 0 = vocab size)
    float   top_p;                  // top-p sampling (1.0 = disabled)
    float   min_p;                  // min-p sampling (0.0 = disabled)
    float   xtc_probability;        // XTC probability (0.0 = disabled)
    float   xtc_threshold;          // XTC threshold (> 0.5 disables XTC)
    float   typ_p;                  // typical-p (1.0 = disabled)
    float   temp;                   // temperature (<= 0.0 = greedy)
    float   dynatemp_range;         // dynamic temperature range (0.0 = disabled)
    float   dynatemp_exponent;      // dynamic temperature exponent

    int32_t penalty_last_n;         // last n tokens to penalize (0 = disable, -1 = context)
    float   penalty_repeat;         // repetition penalty (1.0 = disabled)
    float   penalty_freq;           // frequency penalty (0.0 = disabled)
    float   penalty_present;        // presence penalty (0.0 = disabled)

    float   dry_multiplier;         // DRY multiplier (0.0 = disabled)
    float   dry_base;               // DRY base (1.75 typical)
    int32_t dry_allowed_length;     // DRY allowed length before penalty
    int32_t dry_penalty_last_n;     // DRY tokens to scan (-1 = context)
    std::vector<std::string> dry_sequence_breakers;

    float   adaptive_target;        // adaptive sampling target (-1 = disabled)
    float   adaptive_decay;         // adaptive sampling decay

    int32_t mirostat;               // Mirostat (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
    float   top_n_sigma;            // top-n-sigma (-1.0 = disabled)
    float   mirostat_tau;           // Mirostat tau (target entropy)
    float   mirostat_eta;           // Mirostat eta (learning rate)

    bool    ignore_eos;             // ignore EOS tokens
    bool    no_perf;                // disable performance metrics
    bool    timing_per_token;       // timing per token

    // Grammar settings
    std::string grammar;            // grammar string (GBNF)
    bool    grammar_lazy;           // lazy grammar application
    std::vector<std::string> grammar_triggers; // grammar trigger strings
    std::vector<llama_token> preserved_tokens; // tokens to preserve

    // Logit bias
    std::vector<llama_logit_bias> logit_bias;
    std::vector<llama_logit_bias> logit_bias_eog;

    // Generation prompt for grammar prefill
    std::string generation_prompt;

    // Reasoning budget
    int32_t   reasoning_budget_tokens;
    std::vector<llama_token> reasoning_budget_start;
    std::vector<llama_token> reasoning_budget_end;
    std::vector<llama_token> reasoning_budget_forced;
    std::string              reasoning_budget_message;

    bool    backend_sampling;       // use backend sampling

} mtmd_sampling_params_t;

// ============================================================================
// Speculative Decoding Parameters
// ============================================================================
typedef struct {
    int32_t n_max;                  // max draft tokens
    int32_t n_min;                  // min draft tokens
    float   p_split;                // split probability
    float   p_min;                  // min split probability
    std::string draft_model_path;   // path to draft model
    int32_t n_gpu_layers_draft;     // GPU layers for draft model
} mtmd_speculative_params_t;

// ============================================================================
// Model Parameters
// ============================================================================
typedef struct {
    std::string path;               // model path
    std::string mmproj_path;        // multimodal projector path
    bool    mmproj_use_gpu;         // use GPU for mmproj
    bool    no_mmproj;              // disable multimodal
} mtmd_model_params_t;

// ============================================================================
// Prompt Parameters (for chat-style inference)
// ============================================================================
typedef struct {
    std::string system_prompt;      // system prompt (optional, empty = no system prompt)
    std::string user_prompt;        // user prompt (required)
    bool        add_ass_prefix;     // add "<|start_header_id|>assistant<|end_header_id|>" prefix
} mtmd_prompt_params_t;

// ============================================================================
// Image Data (bitmap in memory)
// ============================================================================
typedef struct {
    uint32_t       width;           // image width
    uint32_t       height;          // image height
    const uint8_t* data;            // image data in RGB format (width * height * 3 bytes)
} mtmd_image_data_t;

// ============================================================================
// Main Inference Parameters (comprehensive, matching mtmd-cli)
// ============================================================================
typedef struct {
    // Model settings
    mtmd_model_params_t model;

    // Backend/device settings
    std::string device;             // device string (e.g., "GPU0", "HTP0,HTP1,HTP2,HTP3", "CPU")
    std::string library_directory;  // directory containing backend .so files (Android)
    int32_t n_gpu_layers;           // number of layers to offload (0 = CPU, 999 = all GPU/NPU)

    // Context settings
    int32_t n_ctx;                  // context size (0 = use model default)
    int32_t n_batch;                // batch size for prompt processing
    int32_t n_ubatch;               // physical batch size

    // Generation settings
    int32_t n_predict;              // max new tokens (-1 = unlimited, -2 = n_ctx - n_prompt)
    int32_t n_keep;                 // number of tokens to keep from initial prompt
    int32_t n_draft;                // number of tokens to draft (speculative decoding)

    // CPU parameters
    mtmd_cpu_params_t cpuparams;
    mtmd_cpu_params_t cpuparams_batch;

    // Sampling parameters
    mtmd_sampling_params_t sampling;

    // Speculative decoding
    mtmd_speculative_params_t speculative;

    // Multimodal settings
    int32_t image_min_tokens;       // minimum image tokens
    int32_t image_max_tokens;       // maximum image tokens
    bool    no_mmproj_offload;      // disable mmproj GPU offload

    // Performance/logging
    bool    use_color;              // use color output
    bool    special;                // enable special token output
    bool    verbose_prompt;         // print prompt tokens
    bool    display_prompt;         // print prompt before generation
    bool    no_perf;                // disable performance metrics
    bool    show_timings;           // show timing on CLI
    bool    warmup;                 // perform warmup run

    // Memory/settings
    bool    use_mmap;               // use memory mapping
    bool    use_mlock;              // use memory lock
    bool    no_kv_offload;          // disable KV offloading
    bool    swa_full;               // full SWA cache
    bool    kv_unified;             // unified KV cache

    // Flash attention (0 = disabled, 1 = enabled, 2 = auto)
    int32_t flash_attn_type;        // flash attention type

    // Chat template
    std::string chat_template;      // custom chat template
    bool    use_jinja;              // use Jinja template engine

    // Chat history persistence (default: false = clear after each inference)
    bool    keep_chat_history;      // keep chat history across inference calls

    // Backend init
    bool    backend_init;           // initialize backend (default: true)
    bool    print_system_info;      // print system info

    // Debug logging (Android only)
    bool    debug_logs_enabled;     // enable debug logs (default: false)
    bool    info_logs_enabled;      // enable info logs (default: false)

} mtmd_inference_params_t;

// ============================================================================
// Inference Result
// ============================================================================
typedef struct {
    std::string text;               // generated text
    int32_t   n_tokens;             // number of tokens generated
    int32_t   n_prompt_tokens;      // number of prompt tokens (including image tokens)
    double    load_time_ms;         // model load time (if applicable)
    double    prompt_eval_time_ms;  // prompt evaluation time
    double    eval_time_ms;         // generation time
    double    tokens_per_second;    // generation throughput
} mtmd_inference_result_t;

// ============================================================================
// Callback Types
// ============================================================================
typedef void (*mtmd_progress_callback_t)(const char* message, void* user_data);
typedef void (*mtmd_token_callback_t)(const char* token, void* user_data);

// ============================================================================
// API Functions
// ============================================================================

/**
 * Initialize the inference engine.
 *
 * This function:
 * 1. Sets environment variables (LD_LIBRARY_PATH, D, GGML_HEXAGON_NDEV)
 * 2. Loads all backends (ggml_backend_load_all)
 * 3. Initializes llama backend
 * 4. Loads the model and mmproj
 * 5. Creates llama context
 * 6. Initializes mtmd context
 * 7. Sets up sampler
 *
 * @param params        Inference parameters (see mtmd_inference_params_t)
 * @param ctx_out       Output pointer for inference context
 * @param progress_cb   Optional progress callback
 * @param user_data     User data passed to progress callback
 * @return              MTMD_API_SUCCESS on success, error code on failure
 */
mtmd_api_status mtmd_inference_init(
    const mtmd_inference_params_t& params,
    mtmd_inference_context_t** ctx_out,
    mtmd_progress_callback_t progress_cb,
    void* user_data
);

/**
 * Run inference with the given image(s) and prompts.
 *
 * This function:
 * 1. Creates bitmap from image data in memory
 * 2. Tokenizes the prompt with image embeddings
 * 3. Evaluates the prompt through the model
 * 4. Generates response tokens
 * 5. Returns the generated text
 *
 * The prompt is constructed as:
 *   [system_prompt] + [user_prompt with <__media__> marker] + [assistant prefix]
 *
 * @param ctx           Inference context (must be initialized)
 * @param images        Array of image data in memory (RGB format)
 * @param n_images      Number of images
 * @param prompt        Prompt parameters (system + user prompts)
 * @param result_out    Output pointer for inference result
 * @param token_cb      Optional token callback for streaming (called for each token)
 * @param user_data     User data passed to token callback
 * @return              MTMD_API_SUCCESS on success, error code on failure
 */
mtmd_api_status mtmd_inference_run(
    mtmd_inference_context_t* ctx,
    const mtmd_image_data_t* images,
    int32_t n_images,
    const mtmd_prompt_params_t& prompt,
    mtmd_inference_result_t* result_out,
    mtmd_token_callback_t token_cb,
    void* user_data
);

/**
 * Free inference context and release resources.
 *
 * @param ctx   Inference context to free
 */
void mtmd_inference_free(mtmd_inference_context_t* ctx);

/**
 * Get error message for status code.
 *
 * @param status    Status code
 * @return          Error message string
 */
const char* mtmd_api_status_string(mtmd_api_status status);

// ============================================================================
// Helper Functions for Default Parameters
// ============================================================================

/**
 * Get default inference parameters.
 *
 * @param params  Pointer to params structure to initialize
 */
void mtmd_inference_params_default(mtmd_inference_params_t* params);

/**
 * Get default sampling parameters.
 *
 * @param params  Pointer to params structure to initialize
 */
void mtmd_sampling_params_default(mtmd_sampling_params_t* params);

/**
 * Get default CPU parameters.
 *
 * @param params  Pointer to params structure to initialize
 */
void mtmd_cpu_params_default(mtmd_cpu_params_t* params);

#ifdef __cplusplus
}
#endif

#endif // MTMD_INFERENCE_API_H
