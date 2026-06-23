// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#include "config.hpp"
#include "json_parser.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace screen_vlm {

bool Config::s_initialized = false;

// Runtime configuration - use static members directly
std::string Config::s_model = "unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL";
std::string Config::s_api_url = "http://192.168.1.46:8080/v1/chat/completions";
std::string Config::s_system_prompt = "You're talking to someone looking at this screen right now. Tell them casually what they're looking at - like you're explaining it to a friend. Just describe the main thing in plain English, no formatting or lists.";
std::string Config::s_user_prompt = "What's on this screen right now? What should the person see? Tell them what matters - keep it casual like you're explaining it to a friend.";
int32_t Config::s_capture_cooldown = 3;
int32_t Config::s_max_tokens = 1024;
double Config::s_temperature = 0.9;
double Config::s_top_p = 0.9;
int32_t Config::s_image_quality = 85;
int32_t Config::s_image_scale = 4;
int32_t Config::s_tts_rate_analysis = 140;
int32_t Config::s_tts_rate_result = 180;

void Config::initialize(const std::string& config_path) {
    if (s_initialized) {
        return;
    }

    // Determine config file path
    std::string path = config_path;
    if (path.empty()) {
        // Try to find config.json in parent directories
        try {
            auto exe_path = fs::current_path();
            // Check current directory first
            if (fs::exists("config.json")) {
                path = "config.json";
            } else if (fs::exists("../config.json")) {
                path = "../config.json";
            } else if (fs::exists("../../config.json")) {
                path = "../../config.json";
            }
        } catch (...) {
            // Fall back to current directory
            path = "config.json";
        }
    }

    // Load from file if it exists
    if (fs::exists(path)) {
        try {
            auto j = load_json_from_file(path);

            if (j.contains("model")) s_model = j["model"].get<std::string>();
            if (j.contains("api_url")) s_api_url = j["api_url"].get<std::string>();
            if (j.contains("system_prompt")) s_system_prompt = j["system_prompt"].get<std::string>();
            if (j.contains("user_prompt")) s_user_prompt = j["user_prompt"].get<std::string>();
            if (j.contains("capture_cooldown")) s_capture_cooldown = j["capture_cooldown"].get<int32_t>();
            if (j.contains("max_tokens")) s_max_tokens = j["max_tokens"].get<int32_t>();
            if (j.contains("temperature")) s_temperature = j["temperature"].get<double>();
            if (j.contains("top_p")) s_top_p = j["top_p"].get<double>();
            if (j.contains("image_quality")) s_image_quality = j["image_quality"].get<int32_t>();
            if (j.contains("image_scale")) s_image_scale = j["image_scale"].get<int32_t>();
            if (j.contains("tts_rate_analysis")) s_tts_rate_analysis = j["tts_rate_analysis"].get<int32_t>();
            if (j.contains("tts_rate_result")) s_tts_rate_result = j["tts_rate_result"].get<int32_t>();

            std::cout << "Config loaded from: " << path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "WARNING: Failed to load config.json: " << e.what() << std::endl;
            std::cerr << "Using default configuration." << std::endl;
        }
    } else {
        std::cout << "No config.json found. Using defaults." << std::endl;
    }

    s_initialized = true;
}

std::string Config::get_model() { return s_model; }
std::string Config::get_api_url() { return s_api_url; }
std::string Config::get_system_prompt() { return s_system_prompt; }
std::string Config::get_user_prompt() { return s_user_prompt; }
int32_t Config::get_capture_cooldown() { return s_capture_cooldown; }
int32_t Config::get_max_tokens() { return s_max_tokens; }
double Config::get_temperature() { return s_temperature; }
double Config::get_top_p() { return s_top_p; }
int32_t Config::get_image_quality() { return s_image_quality; }
int32_t Config::get_image_scale() { return s_image_scale; }
int32_t Config::get_tts_rate_analysis() { return s_tts_rate_analysis; }
int32_t Config::get_tts_rate_result() { return s_tts_rate_result; }

} // namespace screen_vlm
