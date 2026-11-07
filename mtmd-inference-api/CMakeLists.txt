# mtmd-inference-api - Multimodal Inference API Library
# Provides clean API for multimodal inference without console interaction

# Set RPATH to use $ORIGIN (library in same directory as executable)
# Use string() to construct literal $ORIGIN since CMake interprets $ as variable
string(ASCII 36 DOLLAR_SIGN)
set(CMAKE_BUILD_RPATH "${DOLLAR_SIGN}ORIGIN")
set(CMAKE_INSTALL_RPATH "${DOLLAR_SIGN}ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Library sources
set(MTMD_INFERENCE_API_SOURCES
    mtmd-inference-api.cpp
)

set(MTMD_INFERENCE_API_HEADERS
    mtmd-inference-api.h
)

# Create static library
add_library(mtmd-inference-api STATIC
    ${MTMD_INFERENCE_API_SOURCES}
    ${MTMD_INFERENCE_API_HEADERS}
)

# Include directories
target_include_directories(mtmd-inference-api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Link against llama.cpp libraries
target_link_libraries(mtmd-inference-api
    PRIVATE
        llama
        mtmd
        ggml
        llama-common
        Threads::Threads
)

# Newer llama.cpp replaced use_mmap/use_mlock with load_mode; older tags still build without this
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../include/llama.h" LLAMA_H)
if(LLAMA_H MATCHES "enum llama_load_mode")
    target_compile_definitions(mtmd-inference-api PRIVATE MTMD_HAS_LOAD_MODE)
endif()

# Newer llama.cpp reads mtmd_input_text.text_len; older tags do not have the field
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../mtmd/mtmd.h" MTMD_H)
if(MTMD_H MATCHES "size_t text_len")
    target_compile_definitions(mtmd-inference-api PRIVATE MTMD_HAS_TEXT_LEN)
endif()

# Newer llama.cpp selects the vision encoder device through mtmd_context_params.device;
# older tags use the MTMD_BACKEND_DEVICE environment variable
if(MTMD_H MATCHES "ggml_backend_dev_t device")
    target_compile_definitions(mtmd-inference-api PRIVATE MTMD_HAS_CTX_DEVICE)
endif()

# Android logging
if(ANDROID)
    target_link_libraries(mtmd-inference-api PRIVATE log)
endif()

# Set C++ standard
target_compile_features(mtmd-inference-api PUBLIC cxx_std_17)

# Install targets
install(TARGETS mtmd-inference-api
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES mtmd-inference-api.h
    DESTINATION include
)

# Test executable - comprehensive test for device deployment
add_executable(mtmd-inference-test test.cpp)
target_link_libraries(mtmd-inference-test PRIVATE mtmd-inference-api)

# Android logging for test executable
if(ANDROID)
    target_link_libraries(mtmd-inference-test PRIVATE log)
endif()
target_include_directories(mtmd-inference-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mtmd
)
target_compile_features(mtmd-inference-test PRIVATE cxx_std_17)

# Set RPATH using $ORIGIN via ASCII escape to avoid CMake variable interpretation
set_target_properties(mtmd-inference-test PROPERTIES
    LINK_FLAGS "-Wl,-rpath,${DOLLAR_SIGN}ORIGIN"
)

# Profile test executable - profiling test for performance analysis
add_executable(mtmd-profile-test ${CMAKE_CURRENT_SOURCE_DIR}/../../../profiler/mtmd-profile-test.cpp)
target_link_libraries(mtmd-profile-test PRIVATE mtmd-inference-api)

# Android logging for profile test executable
if(ANDROID)
    target_link_libraries(mtmd-profile-test PRIVATE log)
endif()
target_include_directories(mtmd-profile-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mtmd
)
target_compile_features(mtmd-profile-test PRIVATE cxx_std_17)

# Set RPATH using $ORIGIN via ASCII escape to avoid CMake variable interpretation
set_target_properties(mtmd-profile-test PROPERTIES
    LINK_FLAGS "-Wl,-rpath,${DOLLAR_SIGN}ORIGIN"
)