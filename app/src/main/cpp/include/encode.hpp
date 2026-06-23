// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#ifndef ENCODE_HPP
#define ENCODE_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace screen_vlm {

struct ImageData {
    int32_t width;
    int32_t height;
    std::vector<uint8_t> pixels;  // RGB format
};

class ImageEncoder {
public:
    // Resize image to 1/4 dimensions using bilinear interpolation
    // Returns new ImageData with resized pixels
    static ImageData resize_bilinear(const ImageData& input, int32_t scale_factor);

    // Encode ImageData to JPEG bytes
    // quality: 1-100 (Python default: 85)
    // Returns JPEG-encoded byte buffer
    static std::vector<uint8_t> encode_jpeg(const ImageData& image, int32_t quality);

    // Encode ImageData directly to base64 string
    // Resize to 1/4, encode as JPEG quality 85, then base64
    // Matches Python encode_image_to_base64() behavior
    static std::string image_to_base64(const ImageData& image);

    // Base64 encode raw bytes
    static std::string base64_encode(const std::vector<uint8_t>& data);

private:
    // Bilinear interpolation helper
    static uint8_t bilinear_sample(
        const std::vector<uint8_t>& src,
        int32_t src_width, int32_t src_height,
        float x, float y
    );
};

} // namespace screen_vlm

#endif // ENCODE_HPP
