cmake_minimum_required(VERSION 3.22)
project(VlmAnalyze VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android-specific settings
if(ANDROID)
    message(STATUS "Building for Android: ${ANDROID_ABI}")
    add_compile_definitions(ANDROID_PLATFORM __ANDROID__)
endif()

# ============================================================
# Main vlmanalyze library
# ============================================================
add_library(vlmanalyze SHARED
    native-lib.cpp          # JNI bridge
    vlm-impl.cpp            # Android VLM implementation
    android_tts.cpp         # Android TTS JNI

    # Utility CPP sources
    utils/api.cpp
    utils/encode.cpp
    utils/config.cpp
    utils/json_parser.cpp
)

# Include directories
target_include_directories(vlmanalyze PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Android system libraries
target_link_libraries(vlmanalyze
    android          # Android NDK native API
    log              # Android logging (__android_log_print)
    jnigraphics      # Android bitmap/graphics API
    z                # zlib
    OpenSLES         # OpenSL ES for audio (TTS playback)
)

# Find libcurl (provided by Android NDK)
find_library(CURL_LIB curl)
if(CURL_LIB)
    target_link_libraries(vlmanalyze ${CURL_LIB})
    target_compile_definitions(vlmanalyze PRIVATE HAVE_LIBCURL)
    message(STATUS "Found libcurl: ${CURL_LIB}")
else()
    message(STATUS "libcurl not found in NDK, checking alternative names...")
    # Try alternative library name
    find_library(CURL_LIB libcurl)
    if(CURL_LIB)
        target_link_libraries(vlmanalyze ${CURL_LIB})
        target_compile_definitions(vlmanalyze PRIVATE HAVE_LIBCURL)
        message(STATUS "Found libcurl (alternative): ${CURL_LIB}")
    endif()
endif()

# Find libjpeg-turbo
find_library(JPEG_LIB jpeg)
if(JPEG_LIB)
    target_link_libraries(vlmanalyze ${JPEG_LIB})
    target_compile_definitions(vlmanalyze PRIVATE HAVE_LIBJPEG)
    message(STATUS "Found libjpeg")
else()
    message(WARNING "libjpeg not found - JPEG encoding will be limited")
endif()

# Find libpng
find_library(PNG_LIB png)
if(PNG_LIB)
    target_link_libraries(vlmanalyze ${PNG_LIB})
    message(STATUS "Found libpng")
else()
    message(WARNING "libpng not found")
endif()

# C++ standard library
target_link_libraries(vlmanalyze c++_shared)

# Android-specific compile definitions (remove __ANDROID_API__ - NDK sets this automatically)
target_compile_definitions(vlmanalyze PRIVATE
    ANDROID
)

# Optimization flags for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(vlmanalyze PRIVATE -O3)
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vlmanalyze PRIVATE -g -O0)
endif()

# ============================================================
# llama-server-jni library - JNI wrapper for llama-server
# Only builds for arm64-v8a (prebuilt llama-server-impl is arm64 only)
# ============================================================
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(PREBUILT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a")
    set(LLAMA_SERVER_IMPL_LIB "${PREBUILT_LIB_DIR}/libllama-server-impl.so")

    # Verify the library exists
    if(NOT EXISTS "${LLAMA_SERVER_IMPL_LIB}")
        message(WARNING "libllama-server-impl.so not found at ${LLAMA_SERVER_IMPL_LIB}")
        message(WARNING "Run build-llamacpp-android.sh and copy libraries to jniLibs first")
    endif()

    # server-jni library (JNI wrapper for llama-server)
    add_library(llama-server-jni SHARED
        server-jni.cpp
    )

    target_include_directories(llama-server-jni PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a  # For any headers
    )

    target_link_libraries(llama-server-jni PRIVATE
        ${LLAMA_SERVER_IMPL_LIB}
        vlmanalyze  # For g_debug_logs_enabled symbol
        android
        log
        c++_shared
        dl
    )

    target_compile_definitions(llama-server-jni PRIVATE ANDROID)
endif()

# ============================================================
# mtmd-inference-jni library - JNI wrapper for mtmd-inference-api
# Uses direct mtmd-inference-api for multimodal inference
# Only builds for arm64-v8a (prebuilt libraries are arm64 only)
# ============================================================
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(MTMD_API_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/arm64-v8a")
    set(MTMD_INFERENCE_API_LIB "${MTMD_API_LIB_DIR}/libmtmd-inference-api.a")
    set(MTMD_API_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/mtmd-api/include")
    set(MTMD_API_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/mtmd-api")

    # llama.cpp source directory (relative to app/src/main/cpp)
    set(LLAMA_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp")

    # Verify the library exists
    if(NOT EXISTS "${MTMD_INFERENCE_API_LIB}")
        message(WARNING "libmtmd-inference-api.a not found at ${MTMD_INFERENCE_API_LIB}")
        message(WARNING "Run build-mtmd-inference-api.ps1 to build and copy files")
    endif()

    # mtmd-inference-jni library
    add_library(mtmd-inference-jni SHARED
        mtmd-inference-jni.cpp
    )

    target_include_directories(mtmd-inference-jni PRIVATE
        ${MTMD_API_ROOT}
        ${MTMD_API_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a
        ${LLAMA_CPP_ROOT}/tools
        ${LLAMA_CPP_ROOT}/include
        ${LLAMA_CPP_ROOT}/ggml/include
        ${LLAMA_CPP_ROOT}/vendor
        ${LLAMA_CPP_ROOT}/vendor/stb
        ${LLAMA_CPP_ROOT}/common
        ${LLAMA_CPP_ROOT}/src
    )

    # Prebuilt shared libraries from jniLibs
    set(PREBUILT_LLAMA_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/arm64-v8a")

    target_link_libraries(mtmd-inference-jni PRIVATE
        ${MTMD_INFERENCE_API_LIB}
        ${PREBUILT_LLAMA_LIBS}/libllama.so
        ${PREBUILT_LLAMA_LIBS}/libmtmd.so
        ${PREBUILT_LLAMA_LIBS}/libggml.so
        ${PREBUILT_LLAMA_LIBS}/libggml-base.so
        ${PREBUILT_LLAMA_LIBS}/libggml-cpu.so
        ${PREBUILT_LLAMA_LIBS}/libllama-common.so
        vlmanalyze  # For g_debug_logs_enabled symbol
        android
        log
        c++_shared
        dl
    )

    # Set RPATH to use $ORIGIN so libraries find each other in the same directory
    # Use string() to construct literal $ORIGIN since CMake interprets $ as variable
    string(ASCII 36 DOLLAR_SIGN)
    set_target_properties(mtmd-inference-jni PROPERTIES
        LINK_FLAGS "-Wl,-rpath,${DOLLAR_SIGN}ORIGIN"
    )

    target_compile_definitions(mtmd-inference-jni PRIVATE ANDROID)
endif()

# ============================================================
# Note: json_parser.cpp (root) is NOT included in the build.
# It is a duplicate of utils/json_parser.cpp (same functionality).
#
# Note: native-helper.cpp is NOT included in the build.
# This file provided asset extraction for the old "binary execution" approach.
# The current implementation uses JNI libraries (mtmd-inference-jni.cpp) instead.
# ============================================================
