// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#include "api.hpp"
#include "config.hpp"
#include "json_parser.hpp"
#include "debug-flags.h"

#include <android/log.h>

#define LOGI(...) do { if (g_info_logs_enabled) __android_log_print(ANDROID_LOG_INFO, "VLM-Api", __VA_ARGS__); } while(0)
#define LOGD(...) do { if (g_debug_logs_enabled) __android_log_print(ANDROID_LOG_DEBUG, "VLM-Api", __VA_ARGS__); } while(0)

#include <iostream>
#include <sstream>
#include <chrono>
#include <regex>
#include <thread>
#include <cstring>

#ifdef HAVE_LIBCURL
#include <curl/curl.h>
#endif

namespace screen_vlm {

// Simple token counter (estimates tokens similar to tiktoken cl100k_base)
// 1 token ~= 4 characters for English text
static int32_t count_tokens_tiktoken(const std::string& text) {
    if (text.empty()) return 0;
    // Rough estimate: 1 token ~= 4 characters
    // This matches Python's fallback when tiktoken fails
    return static_cast<int32_t>(text.length() / 4);
}

// Stream data structure for libcurl streaming callback
struct StreamData {
    VlmApi::StreamCallback callback;
    std::string buffer;
    std::string accumulated_content;  // Full response content
    int32_t prompt_tokens;
    int32_t completion_tokens;
};

// Write callback for libcurl
static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::string* response = static_cast<std::string*>(userdata);
    response->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

// Streaming write callback for libcurl
// Note: Function renamed from stream_callback to avoid confusion with CURLOPT_WRITEFUNCTION
static size_t streaming_write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    StreamData* data = static_cast<StreamData*>(userdata);
    data->buffer.append(static_cast<char*>(ptr), size * nmemb);

    // Process complete lines
    size_t pos;
    while ((pos = data->buffer.find('\n')) != std::string::npos) {
        std::string line = data->buffer.substr(0, pos);
        data->buffer.erase(0, pos + 1);

        // Trim whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }

        if (line.empty()) continue;

        // Parse SSE line (starts with "data: ")
        if (line.substr(0, 6) == "data: ") {
            std::string data_content = line.substr(6);

            if (data_content == "[DONE]") {
                continue;
            }

            try {
                auto j = json::json::parse(data_content);

                // Update token counts FIRST (before checking content)
                // API may send usage at top level OR inside choices[0]
                if (j.contains("usage")) {
                    data->prompt_tokens = j["usage"].value("prompt_tokens", 0);
                    data->completion_tokens = j["usage"].value("completion_tokens", 0);
                } else if (j.contains("choices") && !j["choices"].empty() &&
                           j["choices"][0].contains("usage")) {
                    // Some APIs send usage inside choices[0]
                    data->prompt_tokens = j["choices"][0]["usage"].value("prompt_tokens", 0);
                    data->completion_tokens = j["choices"][0]["usage"].value("completion_tokens", 0);
                }

                if (data->prompt_tokens > 0 || data->completion_tokens > 0) {
                    std::cout << "\n[Tokens: prompt=" << data->prompt_tokens
                              << " completion=" << data->completion_tokens << "]" << std::flush;
                }

                if (j.contains("choices") && !j["choices"].empty()) {
                    // Match Python: get content from delta
                    std::string content = j["choices"][0].value("delta", json::json::object()).value("content", "");
                    if (!content.empty()) {
                        // Accumulate content (matching Python: full_response += content)
                        data->accumulated_content += content;

                        // Print chunk to console (matching Python: sys.stdout.write)
                        std::cout << content << std::flush;

                        // Call user-provided callback if available
                        if (data->callback) {
                            data->callback(content, data->prompt_tokens + data->completion_tokens);
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Skip invalid JSON lines (matching Python: continue on JSONDecodeError)
            }
        }
    }

    return size * nmemb;
}

std::string VlmApi::build_payload_json(
    const std::string& base64_image,
    const std::string& model,
    const std::string& system_prompt,
    const std::string& user_prompt
) {
    // Generate request ID
    std::stringstream ss;
    ss << "test_" << std::this_thread::get_id();
    std::string request_id = ss.str();

    json::json payload;
    payload["model"] = model;
    payload["stream"] = true;  // Match Python: use streaming
    payload["max_tokens"] = Config::get_max_tokens();
    payload["temperature"] = Config::get_temperature();
    payload["top_p"] = Config::get_top_p();
    payload["request_id"] = request_id;

    json::json messages = json::json::array();

    // System message
    json::json system_msg;
    system_msg["role"] = "system";
    system_msg["content"] = system_prompt;
    messages.push_back(system_msg);

    // User message - use text-only if no image, otherwise include image
    json::json user_msg;
    user_msg["role"] = "user";

    if (base64_image.empty()) {
        // Text-only payload (for testing)
        user_msg["content"] = user_prompt;
    } else {
        // Multi-modal payload with image
        json::json content = json::json::array();
        json::json text_item;
        text_item["type"] = "text";
        text_item["text"] = user_prompt;
        content.push_back(text_item);

        json::json image_item;
        image_item["type"] = "image_url";
        image_item["image_url"]["url"] = "data:image/jpeg;base64," + base64_image;
        content.push_back(image_item);

        user_msg["content"] = content;
    }

    messages.push_back(user_msg);

    payload["messages"] = messages;

    return payload.dump(2);  // Pretty print for comparison
}

ApiResponse VlmApi::analyze_image(
    const std::string& base64_image,
    const std::string& model,
    const std::string& system_prompt,
    const std::string& user_prompt,
    StreamCallback stream_callback
) {
    ApiResponse result;
    result.success = false;
    result.prompt_tokens = 0;
    result.completion_tokens = 0;
    result.total_time = 0.0;

    // Build payload
    std::string payload_json = build_payload_json(base64_image, model, system_prompt, user_prompt);

    // Save payload for test vector comparison
    try {
        auto j = json::json::parse(payload_json);
        save_json_to_file("test_vectors/cpp_api_payload.json", j);
    } catch (...) {
        // Ignore save errors
    }

#ifdef HAVE_LIBCURL
    // Initialize libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        result.error_message = "Failed to initialize libcurl";
        std::cerr << "ERROR: " << result.error_message << std::endl;
        return result;
    }

    auto start = std::chrono::high_resolution_clock::now();

    curl_easy_setopt(curl, CURLOPT_URL, Config::get_api_url().c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_json.c_str());

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Always use streaming mode (matching Python)
    // Use stream_callback for processing SSE data, call user callback if provided
    struct StreamData stream_data;
    stream_data.callback = stream_callback;
    stream_data.prompt_tokens = 0;
    stream_data.completion_tokens = 0;
    stream_data.buffer = "";
    stream_data.accumulated_content = "";

    // Use our internal streaming_write_callback for processing SSE data
    // This matches Python's iter_lines() approach for streaming SSE responses
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, streaming_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, static_cast<void*>(&stream_data));

    // Get HTTP response code for debugging
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        result.error_message = curl_easy_strerror(res);
        std::cerr << "[CURL Error: " << result.error_message << "]" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(end - start).count();

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // Check HTTP status code
    if (http_code >= 400) {
        result.error_message = "HTTP error: " + std::to_string(http_code);
        std::cerr << "[HTTP Error: " << http_code << "]" << std::endl;
        return result;
    }

    if (!result.error_message.empty()) {
        std::cerr << "ERROR: " << result.error_message << std::endl;
        return result;
    }

    // Get accumulated content from streaming
    result.content = stream_data.accumulated_content;
    result.success = !result.content.empty();
    result.prompt_tokens = stream_data.prompt_tokens;
    result.completion_tokens = stream_data.completion_tokens;

    // Fallback: use tiktoken if API didn't provide usage (matching Python)
    if (result.prompt_tokens == 0 && result.completion_tokens == 0) {
        // Use tiktoken for accurate token counting
        int32_t prompt_count = count_tokens_tiktoken(system_prompt + user_prompt);
        int32_t completion_count = count_tokens_tiktoken(result.content);
        result.prompt_tokens = prompt_count;
        result.completion_tokens = completion_count;
        std::cout << "\n[Tokens estimated via tiktoken: prompt=" << prompt_count
                  << " completion=" << completion_count << "]" << std::flush;
    }

#else
    // Mock mode (no libcurl)
    std::cout << "[MOCK MODE - libcurl not available]" << std::endl;
    std::cout << "Would send payload to: " << Config::get_api_url() << std::endl;
    std::cout << "Payload:" << std::endl << payload_json << std::endl;

    result.error_message = "Mock mode - no HTTP client available";
#endif

    return result;
}

// parse_streaming_response is a static method - define it in the namespace
std::string VlmApi::parse_streaming_response(
    const std::string& response_text,
    int32_t& prompt_tokens,
    int32_t& completion_tokens
) {
    std::string content;
    prompt_tokens = 0;
    completion_tokens = 0;

    try {
        auto j = json::json::parse(response_text);
        if (j.contains("choices") && !j["choices"].empty()) {
            content = j["choices"][0].value("message", json::json::object()).value("content", "");
        }
        if (j.contains("usage")) {
            prompt_tokens = j["usage"].value("prompt_tokens", 0);
            completion_tokens = j["usage"].value("completion_tokens", 0);
        }
    } catch (const std::exception& e) {
        // Parse error
    }

    return content;
}

} // namespace screen_vlm
