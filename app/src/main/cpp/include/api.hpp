// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#ifndef API_HPP
#define API_HPP

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace screen_vlm {

struct ApiResponse {
    std::string content;
    int32_t prompt_tokens;
    int32_t completion_tokens;
    double total_time;
    bool success;
    std::string error_message;
};

class VlmApi {
public:
    using StreamCallback = std::function<void(const std::string& chunk, int32_t total_tokens)>;

    // Analyze image by sending to llama.cpp API
    // base64_image: JPEG base64 string
    // model: model name from config
    // system_prompt: system prompt from config
    // user_prompt: user prompt from config
    // stream_callback: optional callback for each chunk (nullptr for no streaming)
    // Returns ApiResponse with full content and token counts
    static ApiResponse analyze_image(
        const std::string& base64_image,
        const std::string& model,
        const std::string& system_prompt,
        const std::string& user_prompt,
        StreamCallback stream_callback = nullptr
    );

    // Build the JSON payload (for testing/comparison)
    static std::string build_payload_json(
        const std::string& base64_image,
        const std::string& model,
        const std::string& system_prompt,
        const std::string& user_prompt
    );

private:
    // Parse SSE streaming response
    static std::string parse_streaming_response(
        const std::string& response_text,
        int32_t& prompt_tokens,
        int32_t& completion_tokens
    );
};

} // namespace screen_vlm

#endif // API_HPP
