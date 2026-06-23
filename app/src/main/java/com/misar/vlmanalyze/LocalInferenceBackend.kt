// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.Context
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * LocalInferenceBackend - Direct inference using LocalInferenceService.
 *
 * This backend bypasses the HTTP server and calls llama.cpp inference APIs directly.
 * It uses the LocalInferenceService which wraps the native inference-jni library.
 *
 * Architecture:
 * - LocalInferenceService loads libinference-jni.so
 * - JNI calls inference-wrapper.cpp which uses llama.cpp common/mtmd APIs directly
 * - No HTTP server, no port binding, no inter-process communication
 * - Model stays loaded for multiple analyses (efficient)
 */
class LocalInferenceBackend(private val context: Context) : InferenceBackend {

    private val inferenceService = LocalInferenceService()
    private var currentConfig: InferenceBackend.BackendConfig? = null
    private var isInitialized = false

    // Streaming callback for real-time token updates
    var onTokenStream: ((String) -> Unit)? = null

    companion object {
        private const val TAG = "VLM-LocalInference"
        private const val DEFAULT_N_CTX = 16384
        private const val DEFAULT_N_BATCH = 512
        private const val DEFAULT_VERBOSE = false

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }

    /**
     * Initialize with optional inference parameters.
     *
     * @param config Backend configuration (model path, mmproj path, backend type)
     * @param nCtx Context window size (default: 16384)
     * @param nBatch Batch size (default: 512)
     * @param verbose Enable verbose logging (default: false)
     * @param htpSessions Number of HTP sessions for NPU backend (default: 4)
     * @param imageMinTokens Minimum image tokens (-1 = unlimited)
     * @param topK Top-k sampling (default: 40)
     * @param topP Top-p sampling (default: 0.95)
     * @param temperature Sampling temperature (default: 0.80)
     * @param repetitionPenalty Repetition penalty (default: 1.00)
     */
    fun initialize(
        config: InferenceBackend.BackendConfig,
        nCtx: Int = DEFAULT_N_CTX,
        nBatch: Int = DEFAULT_N_BATCH,
        verbose: Boolean = DEFAULT_VERBOSE,
        htpSessions: Int = 1,
        imageMinTokens: Int = -1,
        topK: Int = 40,
        topP: Float = 0.95f,
        temperature: Float = 0.80f,
        repetitionPenalty: Float = 1.00f
    ): Boolean {
        i("=== LocalInferenceBackend.initialize ===")
        i("  modelPath: ${config.modelPath}")
        i("  mmprojPath: ${config.mmprojPath}")
        i("  backend: ${config.backend}")
        i("  nCtx: $nCtx")
        i("  nBatch: $nBatch")
        i("  verbose: $verbose")
        i("  htpSessions: $htpSessions")
        i("  imageMinTokens: $imageMinTokens")
        i("  topK: $topK")
        i("  topP: $topP")
        i("  temperature: $temperature")
        i("  repetitionPenalty: $repetitionPenalty")
        currentConfig = config

        val backendStr = when (config.backend) {
            Backend.NPU -> "NPU"
            Backend.GPU -> "GPU"
            else -> "CPU"
        }

        val success = inferenceService.initialize(
            modelPath = config.modelPath,
            mmprojPath = config.mmprojPath,
            backend = backendStr,
            nCtx = nCtx,
            nBatch = nBatch,
            nPredict = config.maxTokens,
            verbose = verbose,
            htpSessions = htpSessions,
            imageMinTokens = imageMinTokens,
            topK = topK,
            topP = topP,
            temperature = temperature,
            repetitionPenalty = repetitionPenalty
        )

        isInitialized = success
        i("LocalInferenceBackend initialized: $success")

        return success
    }

    override fun initialize(config: InferenceBackend.BackendConfig): Boolean {
        return initialize(config, DEFAULT_N_CTX, DEFAULT_N_BATCH, DEFAULT_VERBOSE)
    }

    /**
     * Run inference directly without starting a server.
     * The model is already loaded from initialize().
     */
    fun runInference(imageFile: File, prompt: String, maxTokens: Int = 1024): String {
        if (!isInitialized) {
            Log.e(TAG, "Inference not initialized")
            return "Error: Inference not initialized"
        }

        d("Running inference: image=${imageFile.absolutePath}, prompt=$prompt")

        return inferenceService.generateFromImageWithTiming(imageFile, prompt, maxTokens).content
    }

    /**
     * Shutdown and free resources.
     */
    fun shutdown() {
        i("Shutting down LocalInferenceBackend")
        inferenceService.shutdown()
        isInitialized = false
    }

    override suspend fun analyze(imageBytes: ByteArray, prompt: String, systemPrompt: String?): InferenceBackend.InferenceResult {
        if (!isInitialized) {
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Inference backend not initialized. Call initialize() first.",
                timings = null
            )
        }

        try {
            // Decode JPEG bytes to Bitmap, then extract raw RGB bytes
            val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            if (bitmap == null) {
                return InferenceBackend.InferenceResult(
                    success = false,
                    content = "Failed to decode image",
                    timings = null
                )
            }

            val width = bitmap.width
            val height = bitmap.height

            d("Decoded image: ${width}x${height}, converting to RGB")

            // Convert Bitmap to RGB byte array (3 bytes per pixel: R, G, B)
            val rgbBytes = ByteArray(width * height * 3)
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            for (i in pixels.indices) {
                val pixel = pixels[i]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                rgbBytes[i * 3] = r.toByte()
                rgbBytes[i * 3 + 1] = g.toByte()
                rgbBytes[i * 3 + 2] = b.toByte()
            }
            bitmap.recycle()

            d("RGB bytes: ${rgbBytes.size} bytes (${width}x${height})")

            val config = currentConfig ?: throw IllegalStateException("Not initialized")

            // Use polling-based streaming for real-time token updates
            val resultLatch = java.util.concurrent.CountDownLatch(1)
            var finalContent = ""
            var resultPromptTokens = 0
            var resultCompletionTokens = 0
            var resultTotalTimeMs = 0L
            var resultTokensPerSecond = 0.0
            var error: String? = null

            inferenceService.generateFromBytesWithPolling(
                imageData = rgbBytes,
                width = width,
                height = height,
                prompt = prompt,
                maxTokens = config.maxTokens,
                systemPrompt = systemPrompt,
                callback = object : LocalInferenceService.StreamCallback {
                    override fun onToken(token: String) {
                        d("Stream token from inference: ${token.length} chars, starts with: ${token.take(20)}")
                        onTokenStream?.invoke(token)
                    }

                    override fun onComplete(content: String, promptTokens: Int, completionTokens: Int, totalTimeMs: Long, tokensPerSecond: Double) {
                        finalContent = content
                        resultPromptTokens = promptTokens
                        resultCompletionTokens = completionTokens
                        resultTotalTimeMs = totalTimeMs
                        resultTokensPerSecond = tokensPerSecond
                        i("Streaming complete: ${content.length} chars, $completionTokens tokens in ${totalTimeMs}ms (${tokensPerSecond} t/s)")
                        resultLatch.countDown()
                    }

                    override fun onError(err: String) {
                        error = err
                        Log.e(TAG, "Streaming error: $err")
                        resultLatch.countDown()
                    }
                }
            )

            // Wait for streaming to complete (polling runs in background thread)
            resultLatch.await()

            if (error != null) {
                return InferenceBackend.InferenceResult(
                    success = false,
                    content = error!!,
                    timings = null
                )
            }

            // Use native tokensPerSecond from JNI
            val tps = if (resultTokensPerSecond > 0) resultTokensPerSecond else null

            return InferenceBackend.InferenceResult(
                success = true,
                content = finalContent,
                timings = InferenceBackend.InferenceResult.Timings(
                    promptTokens = resultPromptTokens,
                    completionTokens = resultCompletionTokens,
                    promptPerSecond = if (resultPromptTokens > 0 && resultTotalTimeMs > 0) (resultPromptTokens * 1000.0 / resultTotalTimeMs) else null,
                    predictedPerSecond = tps?.toDouble(),
                    totalTimeMs = resultTotalTimeMs
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Inference failed: ${e.message}",
                timings = null
            )
        }
    }

    override suspend fun analyzeRgbBytes(rgbBytes: ByteArray, width: Int, height: Int, prompt: String, systemPrompt: String?): InferenceBackend.InferenceResult {
        if (!isInitialized) {
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Inference backend not initialized. Call initialize() first.",
                timings = null
            )
        }

        if (rgbBytes.isEmpty() || width <= 0 || height <= 0) {
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Invalid RGB image data",
                timings = null
            )
        }

        try {
            d("Running inference on RGB bytes: ${width}x${height}, ${rgbBytes.size} bytes")

            val config = currentConfig ?: throw IllegalStateException("Not initialized")

            // Use polling-based streaming for real-time token updates
            val resultLatch = java.util.concurrent.CountDownLatch(1)
            var finalContent = ""
            var resultPromptTokens = 0
            var resultCompletionTokens = 0
            var resultTotalTimeMs = 0L
            var resultTokensPerSecond = 0.0
            var error: String? = null

            inferenceService.generateFromBytesWithPolling(
                imageData = rgbBytes,
                width = width,
                height = height,
                prompt = prompt,
                maxTokens = config.maxTokens,
                systemPrompt = systemPrompt,
                callback = object : LocalInferenceService.StreamCallback {
                    override fun onToken(token: String) {
                        d("Stream token from inference: ${token.length} chars, starts with: ${token.take(20)}")
                        onTokenStream?.invoke(token)
                    }

                    override fun onComplete(content: String, promptTokens: Int, completionTokens: Int, totalTimeMs: Long, tokensPerSecond: Double) {
                        finalContent = content
                        resultPromptTokens = promptTokens
                        resultCompletionTokens = completionTokens
                        resultTotalTimeMs = totalTimeMs
                        resultTokensPerSecond = tokensPerSecond
                        i("Streaming complete: ${content.length} chars, $completionTokens tokens in ${totalTimeMs}ms (${tokensPerSecond} t/s)")
                        resultLatch.countDown()
                    }

                    override fun onError(err: String) {
                        error = err
                        Log.e(TAG, "Streaming error: $err")
                        resultLatch.countDown()
                    }
                }
            )

            // Wait for completion
            resultLatch.await()

            if (error != null) {
                return InferenceBackend.InferenceResult(
                    success = false,
                    content = error!!,
                    timings = null
                )
            }

            // Use native tokensPerSecond from JNI
            val tps = if (resultTokensPerSecond > 0) resultTokensPerSecond else null

            return InferenceBackend.InferenceResult(
                success = true,
                content = finalContent,
                timings = InferenceBackend.InferenceResult.Timings(
                    promptTokens = resultPromptTokens,
                    completionTokens = resultCompletionTokens,
                    promptPerSecond = if (resultPromptTokens > 0 && resultTotalTimeMs > 0) (resultPromptTokens * 1000.0 / resultTotalTimeMs) else null,
                    predictedPerSecond = tps?.toDouble(),
                    totalTimeMs = resultTotalTimeMs
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "RGB bytes inference failed", e)
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Inference failed: ${e.message}",
                timings = null
            )
        }
    }

    override fun destroy() {
        i("Destroying LocalInferenceBackend")
        shutdown()
    }
}
