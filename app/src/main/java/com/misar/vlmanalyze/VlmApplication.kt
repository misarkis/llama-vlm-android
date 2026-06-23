// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.app.Application
import android.content.Intent
import android.util.Log
import org.opencv.android.OpenCVLoader

/**
 * Application-level singleton to store projection data and local inference backend.
 * This survives across service restarts within the same app process.
 */
class VlmApplication : Application() {

    companion object {
        private const val TAG = "VLM-Application"
        const val LOG_TAG_APP = "VLM-Application"

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (infoLogsEnabled) Log.i(TAG, msg)
        }

        // These are shared across the entire application process
        var pendingResult: Int? = null
            set(value) {
                field = value
                i("pendingResult set to: $value")
            }
        var pendingData: Intent? = null
            set(value) {
                field = value
                i("pendingData set to: ${value != null}")
            }

        // Local inference backend - set by MainActivity when "Start Local" is pressed
        var localInferenceBackend: LocalInferenceBackend? = null
            set(value) {
                field = value
                i("localInferenceBackend set: ${value != null}")
            }

        // Debug logs enabled flag - controlled by user setting in Local Settings
        // JNI code can check this via isDebugLogsEnabled()
        var debugLogsEnabled: Boolean = false
            set(value) {
                field = value
                i("debugLogsEnabled set to: $value")
            }

        // Info logs enabled flag - controlled by user setting in Local Settings
        var infoLogsEnabled: Boolean = false
            set(value) {
                field = value
                i("infoLogsEnabled set to: $value")
            }

        // Pending capture source and backend mode (for camera mode after permission granted)
        var pendingCaptureSource: MainActivity.CaptureSource? = null
        var pendingBackendMode: String? = null

        // Model paths for camera mode (set by MainActivity when configuring local backend)
        var modelPath: String? = null
        var mmprojPath: String? = null
        var backendType: Backend = Backend.CPU

        // JNI-compatible getter for debug logs flag (must be named differently to avoid clash with property getter)
        fun isDebugLogsEnabled(): Boolean = debugLogsEnabled
    }

    override fun onCreate() {
        super.onCreate()
        i("VlmApplication onCreate")
        i("pendingResult preserved: ${pendingResult != null}, pendingData preserved: ${pendingData != null}")

        // Initialize OpenCV at app startup
        if (OpenCVLoader.initLocal()) {
            Log.i(LOG_TAG_APP, "OpenCV initialized successfully at startup")
        } else {
            Log.e(LOG_TAG_APP, "OpenCV initialization failed at startup")
        }
    }
}
