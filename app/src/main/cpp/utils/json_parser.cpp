// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#include "json_parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace screen_vlm {

bool save_json_to_file(const std::string& filepath, const json::json& data) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    file << data.dump(2);  // Pretty print with 2-space indent
    file.close();
    return true;
}

json::json load_json_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return json::json::parse(buffer.str());
}

} // namespace screen_vlm
