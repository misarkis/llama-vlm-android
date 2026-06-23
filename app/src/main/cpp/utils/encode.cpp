// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

#include "encode.hpp"

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#endif

namespace screen_vlm {

// Base64 encoding table
static const char BASE64_CHARS[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string ImageEncoder::base64_encode(const std::vector<uint8_t>& data) {
    std::ostringstream result;
    size_t i = 0;
    uint8_t b0, b1, b2;

    while (i < data.size()) {
        b0 = data[i++];
        b1 = i < data.size() ? data[i++] : 0;
        b2 = i < data.size() ? data[i++] : 0;

        result << BASE64_CHARS[b0 >> 2];
        result << BASE64_CHARS[((b0 & 0x3) << 4) | (b1 >> 4)];
        result << BASE64_CHARS[((b1 & 0xF) << 2) | (b2 >> 6)];
        result << BASE64_CHARS[b2 & 0x3F];
    }

    // Padding
    size_t padding = (3 - (data.size() % 3)) % 3;
    for (size_t j = 0; j < padding; j++) {
        result << '=';
    }

    return result.str();
}

uint8_t ImageEncoder::bilinear_sample(
    const std::vector<uint8_t>& src,
    int32_t src_width, int32_t src_height,
    float x, float y
) {
    // Get integer and fractional parts
    int32_t x0 = static_cast<int32_t>(floorf(x));
    int32_t y0 = static_cast<int32_t>(floorf(y));
    float fx = x - x0;
    float fy = y - y0;

    // Clamp to valid range
    x0 = std::max(0, std::min(x0, src_width - 1));
    int32_t x1 = std::min(x0 + 1, src_width - 1);
    y0 = std::max(0, std::min(y0, src_height - 1));
    int32_t y1 = std::min(y0 + 1, src_height - 1);

    // Get the four corner pixels
    auto get_pixel = [&](int32_t px, int32_t py) -> uint8_t {
        return src[py * src_width * 3 + px * 3];
    };

    uint8_t tl = get_pixel(x0, y0);  // Top-left
    uint8_t tr = get_pixel(x1, y0);  // Top-right
    uint8_t bl = get_pixel(x0, y1);  // Bottom-left
    uint8_t br = get_pixel(x1, y1);  // Bottom-right

    // Bilinear interpolation
    float top = tl * (1 - fx) + tr * fx;
    float bottom = bl * (1 - fx) + br * fx;
    float result = top * (1 - fy) + bottom * fy;

    return static_cast<uint8_t>(roundf(result));
}

ImageData ImageEncoder::resize_bilinear(const ImageData& input, int32_t scale_factor) {
    ImageData output;
    output.width = input.width / scale_factor;
    output.height = input.height / scale_factor;
    output.pixels.resize(output.width * output.height * 3);

    float scale = static_cast<float>(scale_factor);

    // Resize each channel
    for (int32_t y = 0; y < output.height; y++) {
        for (int32_t x = 0; x < output.width; x++) {
            // Source coordinates (scale by factor)
            float src_x = x * scale + 0.5f - 0.5f / scale;
            float src_y = y * scale + 0.5f - 0.5f / scale;

            // Sample each RGB channel
            for (int32_t c = 0; c < 3; c++) {
                auto get_pixel = [&](int32_t px, int32_t py) -> uint8_t {
                    px = std::max(0, std::min(px, input.width - 1));
                    py = std::max(0, std::min(py, input.height - 1));
                    return input.pixels[py * input.width * 3 + px * 3 + c];
                };

                int32_t x0 = static_cast<int32_t>(floorf(src_x));
                int32_t y0 = static_cast<int32_t>(floorf(src_y));
                float fx = src_x - x0;
                float fy = src_y - y0;

                int32_t x1 = std::min(x0 + 1, input.width - 1);
                int32_t y1 = std::min(y0 + 1, input.height - 1);

                x0 = std::max(0, x0);
                y0 = std::max(0, y0);

                uint8_t tl = get_pixel(x0, y0);
                uint8_t tr = get_pixel(x1, y0);
                uint8_t bl = get_pixel(x0, y1);
                uint8_t br = get_pixel(x1, y1);

                float top = tl * (1 - fx) + tr * fx;
                float bottom = bl * (1 - fx) + br * fx;
                float result = top * (1 - fy) + bottom * fy;

                output.pixels[(y * output.width + x) * 3 + c] =
                    static_cast<uint8_t>(roundf(result));
            }
        }
    }

    return output;
}

std::vector<uint8_t> ImageEncoder::encode_jpeg(const ImageData& image, int32_t quality) {
#ifdef HAVE_LIBJPEG
    // Use libjpeg-turbo to encode JPEG
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    cinfo.image_width = image.width;
    cinfo.image_height = image.height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    // Use jpeg_mem_dest which allocates its own buffer
    // We pass nullptr and it will allocate
    unsigned long jpeg_size = 0;
    uint8_t* jpeg_buffer = nullptr;

    // Custom destination: use memory buffer
    // jpeg_mem_dest signature: jpeg_mem_dest(j_compress_ptr, JOCTET **, unsigned long *)
    // The second arg is a pointer to the buffer pointer (for output allocation)
    // For libjpeg-turbo, we can pass nullptr to let it allocate

    // Actually, we need to use the proper API
    // jpeg_mem_dest will allocate the buffer and set jpeg_size
    // We need to pass the address of a pointer that can receive the allocated buffer

    // Simpler approach: use a local buffer and let jpeg_mem_dest manage it
    // This is the standard way to use jpeg_mem_dest
    jpeg_mem_dest(&cinfo, &jpeg_buffer, &jpeg_size);

    jpeg_start_compress(&cinfo, TRUE);

    // Write image data (row by row)
    while (cinfo.next_scanline < image.height) {
        JSAMPROW row_pointer = reinterpret_cast<JSAMPROW>(
            const_cast<uint8_t*>(&image.pixels[cinfo.next_scanline * image.width * 3])
        );
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // Copy from allocated buffer to vector
    std::vector<uint8_t> jpeg_data(jpeg_buffer, jpeg_buffer + jpeg_size);

    // Free the allocated buffer (jpeg_destroy would have done this, but we need the data)
    // In libjpeg-turbo, the buffer is owned by us after this call

    return jpeg_data;
#else
    return std::vector<uint8_t>();
#endif
}

std::string ImageEncoder::image_to_base64(const ImageData& image) {
    // Step 1: Resize to 1/4 dimensions
    ImageData resized = resize_bilinear(image, 4);

#ifdef HAVE_LIBJPEG
    // Step 2: Encode as JPEG with quality 85
    std::vector<uint8_t> jpeg_data = encode_jpeg(resized, 85);
    if (!jpeg_data.empty()) {
        return base64_encode(jpeg_data);
    }
#endif

    // Fallback: Return base64 of raw RGB data (NOT a JPEG!)
    // This is for testing the pipeline structure only
    std::cerr << "WARNING: libjpeg not available - using raw RGB data (not JPEG)" << std::endl;
    return base64_encode(resized.pixels);
}

} // namespace screen_vlm
