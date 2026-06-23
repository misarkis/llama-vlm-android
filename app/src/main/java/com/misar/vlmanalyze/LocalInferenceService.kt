// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.util.Log
import java.io.File

/**
 * LocalInferenceService - Runs VLM inference using mtmd-inference-api JNI.
 *
 * This service uses the mtmd-inference-api library for local multimodal
 * inference with Qwen2-VL models. Direct API integration (no HTTP server).
 */
class LocalInferenceService {

    private var handle: Long = 0
    private var isInitialized = false

    companion object {
        private const val TAG = "LocalInference"
        private const val DEFAULT_N_CTX = 16384
        private const val DEFAULT_N_BATCH = 512
        private const val DEFAULT_N_GPU_LAYERS = 999

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        /**
         * Build HTP device string based on number of sessions.
         * e.g., 4 sessions -> "HTP0,HTP1,HTP2,HTP3"
         */
        fun buildHtpDeviceString(sessions: Int): String {
            val clampedSessions = sessions.coerceIn(1, 4)
            return (0 until clampedSessions).joinToString(",") { "HTP$it" }
        }

        // Load native library
        init {
            try {
                System.loadLibrary("mtmd-inference-jni")
                d("mtmd-inference-jni loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load mtmd-inference-jni: ${e.message}", e)
            }
        }

        // Store application context for native code
        @JvmStatic
        fun setApplicationContext(context: android.content.Context) {
            applicationCtx = context
        }

        private var applicationCtx: android.content.Context? = null

        // Get native library directory
        fun getNativeLibDir(): String {
            return applicationCtx?.applicationInfo?.nativeLibraryDir ?: ""
        }

        // Token callback for streaming (called from native code)
        @JvmStatic
        private fun onTokenReceived(token: String) {
            // This can be overridden by subclasses or used with a listener
            d("Token: $token")
        }
    }

    /**
     * Initialize the inference backend.
     *
     * @param modelPath Path to the model file (.gguf)
     * @param mmprojPath Path to the multimodal projector file (.gguf)
     * @param backend Backend to use: "CPU", "GPU", or "NPU"
     * @return true if initialization succeeded, false otherwise
     */
    fun initialize(
        modelPath: String,
        mmprojPath: String,
        backend: String,
        nCtx: Int = DEFAULT_N_CTX,
        nBatch: Int = DEFAULT_N_BATCH,
        nPredict: Int = 1024,
        verbose: Boolean = false,
        htpSessions: Int = 1,
        // Inference parameters
        imageMinTokens: Int = -1,
        topK: Int = 40,
        topP: Float = 0.95f,
        temperature: Float = 0.80f,
        repetitionPenalty: Float = 1.00f
    ): Boolean {
        d("Initializing: model=$modelPath, mmproj=$mmprojPath, backend=$backend, ctx=$nCtx, batch=$nBatch, nPredict=$nPredict, verbose=$verbose, htpSessions=$htpSessions")
        d("  Inference params: imageMinTokens=$imageMinTokens, topK=$topK, topP=$topP, temperature=$temperature, repetitionPenalty=$repetitionPenalty")

        val nGpuLayers = when (backend) {
            "NPU" -> DEFAULT_N_GPU_LAYERS
            "GPU" -> DEFAULT_N_GPU_LAYERS
            else -> 0
        }

        val device = when (backend) {
            "NPU" -> buildHtpDeviceString(htpSessions)
            "GPU" -> "GPU"
            else -> "CPU"
        }

        // Get the native library directory
        val libDir = getNativeLibDir()
        d("Native lib dir: $libDir")

        try {
            handle = nativeInit(
                modelPath = modelPath,
                mmprojPath = mmprojPath,
                libDir = libDir,
                device = device,
                nGpuLayers = nGpuLayers,
                nCtx = nCtx,
                nBatch = nBatch,
                nPredict = nPredict,
                verbose = verbose,
                imageMinTokens = imageMinTokens,
                topK = topK,
                topP = topP,
                temperature = temperature,
                repetitionPenalty = repetitionPenalty,
                debugLogsEnabled = VlmApplication.debugLogsEnabled,
                infoLogsEnabled = VlmApplication.infoLogsEnabled
            )

            isInitialized = handle != 0L
            if (verbose) {
                d("Initialization ${if (isInitialized) "succeeded" else "failed"}")
            }
            return isInitialized

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}", e)
            return false
        }
    }

    /**
     * Result from inference including timings
     */
    data class InferenceResult(
        val content: String,
        val promptTokens: Int,
        val completionTokens: Int,
        val totalTimeMs: Long
    )

    /**
     * Callback for streaming token updates
     */
    interface StreamCallback {
        fun onToken(token: String)
        fun onComplete(content: String, promptTokens: Int, completionTokens: Int, totalTimeMs: Long, tokensPerSecond: Double)
        fun onError(error: String)
    }

    /**
     * Run multimodal inference on an image file.
     *
     * @param imageFile Path to the image file
     * @param prompt Text prompt for the VLM
     * @param maxTokens Maximum tokens to generate
     * @return InferenceResult with content and timings
     */
    fun generateFromImageWithTiming(imageFile: File, prompt: String, maxTokens: Int = 512): InferenceResult {
        if (!isInitialized || handle == 0L) {
            Log.e(TAG, "Inference not initialized")
            return InferenceResult("Error: Inference not initialized", 0, 0, 0)
        }

        try {
            val startTime = System.currentTimeMillis()
            d("Running multimodal inference: image=${imageFile.absolutePath}, prompt=$prompt")
            val result = nativeGenerate(handle, imageFile.absolutePath, prompt, maxTokens)
            val endTime = System.currentTimeMillis()

            val content = result ?: "Error: Empty response"
            // Use actual token counts from native side
            val promptTokens = nativeGetLastPromptTokens()
            val completionTokens = nativeGetLastCompletionTokens()
            val totalTimeMs = nativeGetLastTotalTimeMs()

            d("Inference complete: ${content.length} chars, $promptTokens prompt + $completionTokens completion tokens, ${totalTimeMs}ms")
            return InferenceResult(content, promptTokens, completionTokens, totalTimeMs)

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            return InferenceResult("Error: ${e.message}", 0, 0, 0)
        }
    }

    /**
     * Run multimodal inference on an image provided as RGB byte array with streaming.
     * The image data should be in RGB format (3 bytes per pixel: R, G, B).
     *
     * @param imageData RGB byte array (width * height * 3 bytes)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param prompt Text prompt for the VLM
     * @param maxTokens Maximum tokens to generate
     * @param systemPrompt System prompt (optional)
     * @param callback StreamCallback for real-time token updates
     */
    fun generateFromBytesWithStreaming(
        imageData: ByteArray,
        width: Int,
        height: Int,
        prompt: String,
        maxTokens: Int = 512,
        systemPrompt: String? = null,
        callback: StreamCallback
    ) {
        if (!isInitialized || handle == 0L) {
            Log.e(TAG, "Inference not initialized")
            callback.onError("Inference not initialized")
            return
        }

        val startTime = System.currentTimeMillis()
        d("Running streaming inference: ${width}x${height}, prompt=$prompt")

        try {
            val result = nativeGenerateFromBytes(handle, imageData, width, height, prompt, maxTokens, systemPrompt)
            val endTime = System.currentTimeMillis()

            val content = result ?: "Error: Empty response"
            // Use actual token counts from native side
            val promptTokens = nativeGetLastPromptTokens()
            val completionTokens = nativeGetLastCompletionTokens()
            val totalTimeMs = nativeGetLastTotalTimeMs()

            val tokensPerSecond = nativeGetLastTokensPerSecond()
            callback.onComplete(content, promptTokens, completionTokens, totalTimeMs, tokensPerSecond)

            d("Streaming complete: ${content.length} chars, $promptTokens prompt + $completionTokens completion tokens, ${totalTimeMs}ms")

        } catch (e: Exception) {
            Log.e(TAG, "Streaming inference failed", e)
            callback.onError("Inference failed: ${e.message}")
        }
    }

    /**
     * Run multimodal inference on an image provided as RGB byte array.
     * The image data should be in RGB format (3 bytes per pixel: R, G, B).
     *
     * @param imageData RGB byte array (width * height * 3 bytes)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param prompt Text prompt for the VLM
     * @param maxTokens Maximum tokens to generate
     * @return InferenceResult with content and timings
     */
    fun generateFromBytesWithTiming(imageData: ByteArray, width: Int, height: Int, prompt: String, maxTokens: Int = 512, systemPrompt: String? = null): InferenceResult {
        if (!isInitialized || handle == 0L) {
            Log.e(TAG, "Inference not initialized")
            return InferenceResult("Error: Inference not initialized", 0, 0, 0)
        }

        try {
            val startTime = System.currentTimeMillis()
            d("Running multimodal inference from bytes: ${width}x${height}, prompt=$prompt, systemPrompt=${systemPrompt ?: "none"}")
            val result = nativeGenerateFromBytes(handle, imageData, width, height, prompt, maxTokens, systemPrompt)

            val content = result ?: "Error: Empty response"
            // Use actual token counts from native side
            val promptTokens = nativeGetLastPromptTokens()
            val completionTokens = nativeGetLastCompletionTokens()
            val totalTimeMs = nativeGetLastTotalTimeMs()

            d("Inference complete: ${content.length} chars, $promptTokens prompt + $completionTokens completion tokens, ${totalTimeMs}ms")
            return InferenceResult(content, promptTokens, completionTokens, totalTimeMs)

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            return InferenceResult("Error: ${e.message}", 0, 0, 0)
        }
    }

    /**
     * Check if the backend is valid/initialized.
     */
    fun isValid(): Boolean {
        return isInitialized && handle != 0L && nativeIsValid(handle)
    }

    /**
     * Shutdown and free resources.
     */
    fun shutdown() {
        if (handle != 0L) {
            d("Shutting down...")
            nativeFree(handle)
            handle = 0L
            isInitialized = false
            d("Shutdown complete")
        }
    }

    /**
     * Register a token callback for streaming responses.
     *
     * @param callbackClass The class containing the callback method
     * @param callbackMethod The method to call for each token
     */
    fun setTokenCallback(callbackClass: Class<*>, callbackMethod: java.lang.reflect.Method) {
        try {
            val methodId = nativeGetMethodId(callbackClass, callbackMethod.name)
            nativeSetTokenCallback(callbackClass, methodId)
            d("Token callback registered")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to register token callback: ${e.message}", e)
        }
    }

    /**
     * Get current stream buffer contents.
     * Call this during inference to poll for new tokens.
     *
     * @return Current accumulated token stream
     */
    fun getStreamBuffer(): String {
        return try {
            nativeGetStreamBuffer()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get stream buffer: ${e.message}")
            ""
        }
    }

    /**
     * Reset the stream buffer.
     * Call this before starting a new inference.
     */
    fun resetStreamBuffer() {
        try {
            nativeResetStreamBuffer()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to reset stream buffer: ${e.message}")
        }
    }

    /**
     * Run multimodal inference with polling-based streaming.
     * This method runs inference in a background thread and polls
     * the shared buffer for real-time token updates.
     * Tokens are accumulated into chunks before callback for smoother TTS.
     *
     * @param imageData RGB byte array
     * @param width Image width
     * @param height Image height
     * @param prompt Text prompt
     * @param maxTokens Maximum tokens to generate
     * @param systemPrompt System prompt (optional)
     * @param callback StreamCallback for real-time updates
     */
    fun generateFromBytesWithPolling(
        imageData: ByteArray,
        width: Int,
        height: Int,
        prompt: String,
        maxTokens: Int = 512,
        systemPrompt: String? = null,
        callback: StreamCallback
    ) {
        if (!isInitialized || handle == 0L) {
            Log.e(TAG, "Inference not initialized")
            callback.onError("Inference not initialized")
            return
        }

        d("Starting polling-based streaming inference: ${width}x${height}")

        // Reset buffer before starting
        resetStreamBuffer()

        val startTime = System.currentTimeMillis()
        var lastBufferLength = 0
        var inferenceDone = false
        var resultContent = ""
        val chunkSize = 80  // Accumulate ~80 chars before callback (about 20-25 words)
        val chunkBuffer = StringBuilder()

        // Run inference + polling in background thread
        Thread {
            try {
                // Start inference in another thread (so we can poll while it runs)
                val inferenceThread = Thread {
                    resultContent = nativeGenerateFromBytes(handle, imageData, width, height, prompt, maxTokens, systemPrompt) ?: ""
                    inferenceDone = true
                }
                inferenceThread.start()

                // Poll the buffer while inference is running
                while (!inferenceDone) {
                    val currentBuffer = getStreamBuffer()
                    if (currentBuffer.length > lastBufferLength) {
                        chunkBuffer.append(currentBuffer.substring(lastBufferLength))
                        lastBufferLength = currentBuffer.length

                        // Only callback when we have enough characters for a smooth chunk
                        if (chunkBuffer.length >= chunkSize) {
                            val chunk = chunkBuffer.toString()
                            d("Sending chunk: ${chunk.length} chars, buffer pos: $lastBufferLength -> ${currentBuffer.length}")
                            chunkBuffer.clear()
                            // Update to current position to avoid re-reading
                            lastBufferLength = currentBuffer.length
                            callback.onToken(chunk)
                        }
                    }
                    Thread.sleep(10)  // Poll every 10ms
                }

                // Wait for inference thread to fully complete
                inferenceThread.join()

                // Check shared buffer one more time for any content we might have missed
                val finalBuffer = getStreamBuffer()
                if (finalBuffer.length > lastBufferLength) {
                    val remaining = finalBuffer.substring(lastBufferLength)
                    d("Sending final chunk from buffer: ${remaining.length} chars")
                    callback.onToken(remaining)
                }

                // Use actual token counts from native side
                val promptTokens = nativeGetLastPromptTokens()
                val completionTokens = nativeGetLastCompletionTokens()
                val totalTimeMs = nativeGetLastTotalTimeMs()
                val tokensPerSecond = nativeGetLastTokensPerSecond()

                // Also send any remaining in chunkBuffer
                if (chunkBuffer.length > 0) {
                    val remaining = chunkBuffer.toString()
                    d("Sending final chunk from chunkBuffer: ${remaining.length} chars")
                    callback.onToken(remaining)
                    chunkBuffer.clear()
                }

                callback.onComplete(resultContent, promptTokens, completionTokens, totalTimeMs, tokensPerSecond)
                d("Polling streaming complete: $promptTokens prompt + $completionTokens completion tokens, ${totalTimeMs}ms")

            } catch (e: Exception) {
                Log.e(TAG, "Polling inference failed", e)
                callback.onError("Inference failed: ${e.message}")
            }
        }.start()
    }

    // Native methods (implemented in mtmd-inference-jni.cpp)
    private external fun nativeInit(
        modelPath: String,
        mmprojPath: String,
        libDir: String,
        device: String,
        nGpuLayers: Int,
        nCtx: Int,
        nBatch: Int,
        nPredict: Int,
        verbose: Boolean,
        imageMinTokens: Int,
        topK: Int,
        topP: Float,
        temperature: Float,
        repetitionPenalty: Float,
        debugLogsEnabled: Boolean,
        infoLogsEnabled: Boolean
    ): Long

    // Get actual token counts from native side (after calling nativeGenerate*)
    private external fun nativeGetLastPromptTokens(): Int
    private external fun nativeGetLastCompletionTokens(): Int
    private external fun nativeGetLastTotalTimeMs(): Long
    private external fun nativeGetLastTokensPerSecond(): Double

    private external fun nativeGenerate(
        handle: Long,
        imagePath: String,
        prompt: String,
        maxTokens: Int
    ): String

    private external fun nativeGenerateFromBytes(
        handle: Long,
        imageData: ByteArray,
        width: Int,
        height: Int,
        prompt: String,
        maxTokens: Int,
        systemPrompt: String?
    ): String

    private external fun nativeFree(handle: Long)

    private external fun nativeIsValid(handle: Long): Boolean

    private external fun nativeSetTokenCallback(callbackClass: Class<*>, methodId: Long)

    private external fun nativeGetMethodId(callbackClass: Class<*>, methodName: String): Long

    private external fun nativeGetTokenCount(): Int

    private external fun nativeGetStreamBuffer(): String

    private external fun nativeResetStreamBuffer()
}
