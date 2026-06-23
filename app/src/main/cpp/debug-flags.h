// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

// debug-flags.h
// Shared debug flags for all native libraries

#ifndef VLM_DEBUG_FLAGS_H
#define VLM_DEBUG_FLAGS_H

// Global debug logs enabled flag - set by Kotlin via JNI in native-lib.cpp
// This is declared extern here and defined in native-lib.cpp
extern bool g_debug_logs_enabled;

// Global info logs enabled flag - set by Kotlin via JNI in native-lib.cpp
extern bool g_info_logs_enabled;

#endif // VLM_DEBUG_FLAGS_H
