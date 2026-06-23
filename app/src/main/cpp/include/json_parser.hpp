// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <string>
#include <nlohmann/json.hpp>

namespace screen_vlm {

// JSON namespace wrapper for nlohmann::json
namespace json = nlohmann;

// Helper function to save JSON to file with pretty printing
bool save_json_to_file(const std::string& filepath, const json::json& data);

// Helper function to load JSON from file
json::json load_json_from_file(const std::string& filepath);

} // namespace screen_vlm

#endif // JSON_PARSER_HPP
