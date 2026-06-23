// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

/**
 * Backend selection for VLM inference.
 *
 * - CPU:  Uses ARM cores, stable but slower
 * - GPU:  Uses Adreno GPU, good performance
 * - NPU:  Uses Qualcomm Hexagon HTP NPU, best performance (experimental)
 */
enum class Backend {
    CPU,
    GPU,
    NPU
}

/**
 * Inference backend interface - implemented by LocalBackend and ServerBackend
 */
interface InferenceBackend {
    data class InferenceResult(
        val success: Boolean,
        val content: String,
        val timings: Timings?
    ) {
        data class Timings(
            val promptTokens: Int,
            val completionTokens: Int,
            val promptPerSecond: Double?,
            val predictedPerSecond: Double?,
            val totalTimeMs: Long = 0
        )
    }

    data class BackendConfig(
        val modelPath: String,
        val mmprojPath: String,
        val backend: Backend,
        val maxTokens: Int = 1024,
        val temperature: Float = 0.9f,
        val topP: Float = 0.9f
    )

    fun initialize(config: BackendConfig): Boolean
    suspend fun analyze(imageBytes: ByteArray, prompt: String, systemPrompt: String? = null): InferenceResult

    /**
     * Analyze using raw RGB byte array (no JPEG encoding).
     * @param rgbBytes RGB byte array (width * height * 3 bytes)
     * @param width Image width
     * @param height Image height
     * @param prompt User prompt
     * @param systemPrompt System prompt (optional)
     */
    suspend fun analyzeRgbBytes(rgbBytes: ByteArray, width: Int, height: Int, prompt: String, systemPrompt: String? = null): InferenceResult

    fun destroy()
}
