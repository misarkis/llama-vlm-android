// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.app.Activity
import android.os.Bundle
import android.util.Log
import java.io.File

/**
 * Automated test activity for NPU backend inference (default).
 *
 * This activity:
 * 1. Copies model/mmproj from /data/local/tmp/llama.cpp/models/Qwen2-VL-2B/ to app storage
 * 2. Copies test image from /data/local/tmp/llama.cpp/images/test-small.png to app storage
 * 3. Initializes NPU backend using mtmd-inference-api JNI
 * 4. Runs multimodal inference on the test image
 * 5. Logs results to logcat
 *
 * Run with: adb shell am start -n com.misar.vlmanalyze/.TestAutoInferenceActivity
 * Check logs with: adb logcat -s VLM-AutoTest:V
 */
class TestAutoInferenceActivity : Activity() {

    private val localInferenceService = LocalInferenceService()

    companion object {
        private const val TAG = "VLM-AutoTest"

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        // Source locations (must be pushed via adb first)
        const val SOURCE_MODEL_DIR = "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B"
        const val SOURCE_IMAGE_PATH = "/data/local/tmp/llama.cpp/images/test-small.png"
        const val SOURCE_CLI_PATH = "/data/local/tmp/llama.cpp/bin/llama-mtmd-cli"
        const val SOURCE_LIB_DIR = "/data/local/tmp/llama.cpp/bin"

        // File names to look for (matching actual files on device)
        const val MODEL_FILENAME = "Qwen2-VL-2B-Q4_K_M.gguf"
        const val MMPROJ_FILENAME = "mmproj-f16.gguf"
        const val TEST_PROMPT = "Describe this image in detail."
        const val MAX_TOKENS = 512
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set application context for native code
        LocalInferenceService.setApplicationContext(applicationContext)

        i( "========================================")
        super.onCreate(savedInstanceState)

        i( "========================================")
        i( "Auto-inference test starting")
        i( "========================================")

        try {
            // Step 1: Copy model files to app storage
            i( "Step 1: Copying model files...")
            val modelPath = copyModelFilesIfNeeded()

            if (modelPath == null) {
                Log.e(TAG, "Failed to copy model files. Check source directory.")
                finishTest(false, "Failed to copy model files")
                return
            }

            i( "Model copied to: $modelPath")

            // Step 2: Copy test image to app storage
            i( "Step 2: Copying test image...")
            val imagePath = copyImageIfNeeded()

            if (imagePath == null) {
                Log.e(TAG, "Failed to copy test image. Check source path.")
                finishTest(false, "Failed to copy test image")
                return
            }

            i( "Image copied to: $imagePath")

            // Step 3: Initialize NPU backend using JNI direct API (default)
            i( "Step 3: Initializing NPU backend via JNI...")

            val mmprojPath = File(filesDir, MMPROJ_FILENAME).absolutePath

            val initSuccess = localInferenceService.initialize(
                modelPath = modelPath,
                mmprojPath = mmprojPath,
                backend = "NPU"
            )

            if (!initSuccess) {
                Log.e(TAG, "Failed to initialize NPU backend via JNI")
                finishTest(false, "JNI backend initialization failed")
                return
            }

            i( "NPU backend initialized successfully")

            // Step 4: Run inference
            i( "Step 4: Running multimodal inference...")
            val imageFile = File(imagePath)

            val result = localInferenceService.generateFromImageWithTiming(
                imageFile = imageFile,
                prompt = TEST_PROMPT,
                maxTokens = MAX_TOKENS
            ).content

            i( "Inference result: ${result.take(200)}...")
            i( "Result length: ${result.length} chars")

            // Cleanup
            localInferenceService.shutdown()

            if (result.startsWith("Error")) {
                finishTest(false, result)
            } else {
                finishTest(true, "Inference succeeded")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Test failed with exception", e)
            finishTest(false, "Exception: ${e.message}")
        }
    }

    /**
     * Copy model files from source to app storage if not already present.
     * Returns the model path if successful, null otherwise.
     */
    private fun copyModelFilesIfNeeded(): String? {
        val modelSrc = File(SOURCE_MODEL_DIR, MODEL_FILENAME)
        val mmprojSrc = File(SOURCE_MODEL_DIR, MMPROJ_FILENAME)

        // Check source files exist
        if (!modelSrc.exists()) {
            Log.e(TAG, "Model not found at: ${modelSrc.absolutePath}")
            Log.e(TAG, "Push model with: adb push <local-model> $SOURCE_MODEL_DIR/")
            return null
        }

        if (!mmprojSrc.exists()) {
            Log.e(TAG, "MMProj not found at: ${mmprojSrc.absolutePath}")
            Log.e(TAG, "Push mmproj with: adb push <local-mmproj> $SOURCE_MODEL_DIR/")
            return null
        }

        // Copy to app private storage
        val modelDest = File(filesDir, MODEL_FILENAME)
        val mmprojDest = File(filesDir, MMPROJ_FILENAME)

        try {
            if (!modelDest.exists()) {
                i( "Copying model (${modelSrc.length()} bytes)...")
                modelSrc.copyTo(modelDest, overwrite = false)
                i( "Model copied to: ${modelDest.absolutePath}")
            } else {
                i( "Model already exists: ${modelDest.absolutePath}")
            }

            if (!mmprojDest.exists()) {
                i( "Copying mmproj (${mmprojSrc.length()} bytes)...")
                mmprojSrc.copyTo(mmprojDest, overwrite = false)
                i( "MMProj copied to: ${mmprojDest.absolutePath}")
            } else {
                i( "MMProj already exists: ${mmprojDest.absolutePath}")
            }

            return modelDest.absolutePath

        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy model files: ${e.message}")
            return null
        }
    }

    /**
     * Copy test image from source to app storage if not already present.
     * Returns the image path if successful, null otherwise.
     */
    private fun copyImageIfNeeded(): String? {
        val imageSrc = File(SOURCE_IMAGE_PATH)

        if (!imageSrc.exists()) {
            Log.e(TAG, "Test image not found at: ${imageSrc.absolutePath}")
            Log.e(TAG, "Push image with: adb push <local-image> $SOURCE_IMAGE_PATH")
            return null
        }

        val imageDest = File(filesDir, "test-image.png")

        try {
            if (!imageDest.exists()) {
                i( "Copying test image (${imageSrc.length()} bytes)...")
                imageSrc.copyTo(imageDest, overwrite = false)
                i( "Test image copied to: ${imageDest.absolutePath}")
            } else {
                i( "Test image already exists: ${imageDest.absolutePath}")
            }

            return imageDest.absolutePath

        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy test image: ${e.message}")
            return null
        }
    }

    /**
     * Finish the test and optionally schedule a restart after delay.
     */
    private fun finishTest(success: Boolean, message: String) {
        i( "========================================")
        i( "TEST ${if (success) "PASSED" else "FAILED"}: $message")
        i( "========================================")

        // Exit the activity
        finish()
    }
}
