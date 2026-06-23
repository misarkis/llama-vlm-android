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