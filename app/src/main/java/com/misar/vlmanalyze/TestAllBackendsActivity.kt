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
 * Test all backends (CPU, GPU, NPU) for multimodal inference.
 *
 * This activity tests each backend sequentially:
 * 1. CPU backend (device="CPU0", nGpuLayers=0)
 * 2. GPU backend (device="GPU0", nGpuLayers=999)
 * 3. NPU backend (device="HTP0,HTP1,HTP2,HTP3", nGpuLayers=999)
 *
 * Run with: adb shell am start -n com.misar.vlmanalyze/.TestAllBackendsActivity
 * Check logs with: adb logcat -s AllBackendTest:V
 */
class TestAllBackendsActivity : Activity() {

    private val localInferenceService = LocalInferenceService()

    companion object {
        private const val TAG = "AllBackendTest"

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        // Source locations (must be pushed via adb first)
        const val SOURCE_MODEL_DIR = "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B"
        const val SOURCE_IMAGE_PATH = "/data/local/tmp/llama.cpp/images/test-small.png"

        // File names
        const val MODEL_FILENAME = "Qwen2-VL-2B-Q4_K_M.gguf"
        const val MMPROJ_FILENAME = "mmproj-f16.gguf"
        const val TEST_PROMPT = "Describe this image in detail."
        const val MAX_TOKENS = 128
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        LocalInferenceService.setApplicationContext(applicationContext)

        i( "========================================")
        i( "Testing all backends: CPU, GPU, NPU")
        i( "========================================")

        try {
            // Copy model files if needed
            val modelPath = copyModelFilesIfNeeded()
            if (modelPath == null) {
                Log.e(TAG, "Failed to copy model files")
                finishTest(false, "Failed to copy model files")
                return
            }

            // Copy test image if needed
            val imagePath = copyImageIfNeeded()
            if (imagePath == null) {
                Log.e(TAG, "Failed to copy test image")
                finishTest(false, "Failed to copy test image")
                return
            }

            val mmprojPath = File(filesDir, MMPROJ_FILENAME).absolutePath
            val imageFile = File(imagePath)

            // Test CPU backend
            i( "========================================")
            i( "TESTING CPU BACKEND")
            i( "========================================")
            val cpuResult = testBackend("CPU", modelPath, mmprojPath, imageFile)
            i( "CPU backend result: $cpuResult")

            // Test GPU backend
            i( "========================================")
            i( "TESTING GPU BACKEND")
            i( "========================================")
            val gpuResult = testBackend("GPU", modelPath, mmprojPath, imageFile)
            i( "GPU backend result: $gpuResult")

            // Test NPU backend
            i( "========================================")
            i( "TESTING NPU BACKEND")
            i( "========================================")
            val npuResult = testBackend("NPU", modelPath, mmprojPath, imageFile)
            i( "NPU backend result: $npuResult")

            // Summary
            i( "========================================")
            i( "BACKEND TEST SUMMARY")
            i( "========================================")
            i( "CPU: $cpuResult")
            i( "GPU: $gpuResult")
            i( "NPU: $npuResult")

            val allPassed = cpuResult && gpuResult && npuResult
            finishTest(allPassed, "All backend tests completed")

        } catch (e: Exception) {
            Log.e(TAG, "Test failed with exception", e)
            finishTest(false, "Exception: ${e.message}")
        }
    }

    /**
     * Test a specific backend.
     * Returns true if the test succeeded, false otherwise.
     */
    private fun testBackend(backend: String, modelPath: String, mmprojPath: String, imageFile: File): Boolean {
        try {
            // Initialize backend
            val initSuccess = localInferenceService.initialize(
                modelPath = modelPath,
                mmprojPath = mmprojPath,
                backend = backend
            )

            if (!initSuccess) {
                Log.e(TAG, "$backend backend initialization failed")
                return false
            }

            i( "$backend backend initialized successfully")

            // Run inference
            val result = localInferenceService.generateFromImageWithTiming(
                imageFile = imageFile,
                prompt = TEST_PROMPT,
                maxTokens = MAX_TOKENS
            ).content

            // Shutdown
            localInferenceService.shutdown()

            if (result.startsWith("Error")) {
                Log.e(TAG, "$backend inference error: $result")
                return false
            }

            i( "$backend inference succeeded: ${result.take(100)}...")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "$backend backend test failed: ${e.message}", e)
            return false
        }
    }

    private fun copyModelFilesIfNeeded(): String? {
        val modelSrc = File(SOURCE_MODEL_DIR, MODEL_FILENAME)
        val mmprojSrc = File(SOURCE_MODEL_DIR, MMPROJ_FILENAME)

        if (!modelSrc.exists() || !mmprojSrc.exists()) {
            Log.e(TAG, "Model files not found at source")
            return null
        }

        val modelDest = File(filesDir, MODEL_FILENAME)
        val mmprojDest = File(filesDir, MMPROJ_FILENAME)

        try {
            if (!modelDest.exists()) {
                modelSrc.copyTo(modelDest)
                i( "Model copied to: ${modelDest.absolutePath}")
            }

            if (!mmprojDest.exists()) {
                mmprojSrc.copyTo(mmprojDest)
                i( "MMProj copied to: ${mmprojDest.absolutePath}")
            }

            return modelDest.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy model files: ${e.message}")
            return null
        }
    }

    private fun copyImageIfNeeded(): String? {
        val imageSrc = File(SOURCE_IMAGE_PATH)

        if (!imageSrc.exists()) {
            Log.e(TAG, "Test image not found at source")
            return null
        }

        val imageDest = File(filesDir, "test-image.png")

        try {
            if (!imageDest.exists()) {
                imageSrc.copyTo(imageDest)
                i( "Test image copied to: ${imageDest.absolutePath}")
            }

            return imageDest.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy test image: ${e.message}")
            return null
        }
    }

    private fun finishTest(success: Boolean, message: String) {
        i( "========================================")
        i( "TEST ${if (success) "PASSED" else "FAILED"}: $message")
        i( "========================================")
        finish()
    }
}
