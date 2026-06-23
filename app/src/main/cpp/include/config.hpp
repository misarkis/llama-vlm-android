// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <cstdint>

namespace screen_vlm {

class Config {
public:
    // Get configuration values (matching Python config.py)
    static std::string get_model();
    static std::string get_api_url();
    static std::string get_system_prompt();
    static std::string get_user_prompt();
    static int32_t get_capture_cooldown();
    static int32_t get_max_tokens();
    static double get_temperature();
    static double get_top_p();
    static int32_t get_image_quality();
    static int32_t get_image_scale();
    static int32_t get_tts_rate_analysis();
    static int32_t get_tts_rate_result();

    // Initialize config from file (call once at startup)
    static void initialize(const std::string& config_path = "");

    // Runtime-configurable members (for Android assets loading)
    static std::string s_model;
    static std::string s_api_url;
    static std::string s_system_prompt;
    static std::string s_user_prompt;
    static int32_t s_capture_cooldown;
    static int32_t s_max_tokens;
    static double s_temperature;
    static double s_top_p;
    static int32_t s_image_quality;
    static int32_t s_image_scale;
    static int32_t s_tts_rate_analysis;
    static int32_t s_tts_rate_result;

private:
    static bool s_initialized;
};

} // namespace screen_vlm

#endif // CONFIG_HPP
