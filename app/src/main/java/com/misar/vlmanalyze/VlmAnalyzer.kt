// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.util.Log
import org.json.JSONObject

/**
 * Kotlin wrapper for VLM analysis functionality.
 * Uses OkHttp for API calls instead of native libcurl.
 */
class VlmAnalyzer {

    private var isInitialized = false
    private val httpClient = HttpClient.instance

    /**
     * Initialize the VLM analyzer.
     */
    fun initialize(storagePath: String): Boolean {
        if (isInitialized) {
            Log.w(TAG, "VlmAnalyzer already initialized")
            return true
        }

        try {
            // Load config from storage or use defaults
            Config.load(storagePath)
            isInitialized = true
            i( "VlmAnalyzer initialized successfully")
            i( "  API URL: ${Config.apiUrl}")
            i( "  Model: ${Config.model}")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize VlmAnalyzer", e)
            return false
        }
    }

    /**
     * Check if the analyzer is ready.
     */
    fun isReady(): Boolean = isInitialized

    /**
     * Analyze an image (base64 encoded) using the VLM API.
     */
    suspend fun analyze(
        base64Image: String,
        systemPrompt: String = Config.systemPrompt,
        userPrompt: String = Config.userPrompt,
        streamCallback: HttpClient.StreamCallback? = null
    ): HttpClient.ApiResult {
        if (!isInitialized) {
            Log.e(TAG, "VlmAnalyzer not initialized")
            return HttpClient.ApiResult(
                success = false,
                content = "",
                promptTokens = 0,
                completionTokens = 0,
                errorMessage = "VlmAnalyzer not initialized"
            )
        }

        try {
            // Build messages
            val messages = listOf(
                mapOf("role" to "system", "content" to systemPrompt),
                mapOf("role" to "user", "content" to listOf(
                    mapOf("type" to "text", "text" to userPrompt),
                    mapOf("type" to "image_url", "image_url" to mapOf(
                        "url" to "data:image/jpeg;base64,$base64Image"
                    ))
                ))
            )

            return httpClient.chatCompletion(
                model = Config.model,
                messages = messages,
                maxTokens = Config.maxTokens,
                temperature = Config.temperature,
                topP = Config.topP,
                streamCallback = streamCallback
            )
        } catch (e: Exception) {
            Log.e(TAG, "Analysis failed", e)
            return HttpClient.ApiResult(
                success = false,
                content = "",
                promptTokens = 0,
                completionTokens = 0,
                errorMessage = e.message
            )
        }
    }

    /**
     * Analyze text-only input (for testing).
     */
    suspend fun analyzeText(
        userPrompt: String,
        systemPrompt: String = Config.systemPrompt
    ): HttpClient.ApiResult {
        if (!isInitialized) {
            Log.e(TAG, "VlmAnalyzer not initialized")
            return HttpClient.ApiResult(
                success = false,
                content = "",
                promptTokens = 0,
                completionTokens = 0,
                errorMessage = "VlmAnalyzer not initialized"
            )
        }

        try {
            val messages = listOf(
                mapOf("role" to "system", "content" to systemPrompt),
                mapOf("role" to "user", "content" to userPrompt)
            )

            return httpClient.chatCompletion(
                model = Config.model,
                messages = messages,
                maxTokens = Config.maxTokens,
                temperature = Config.temperature,
                topP = Config.topP
            )
        } catch (e: Exception) {
            Log.e(TAG, "Text analysis failed", e)
            return HttpClient.ApiResult(
                success = false,
                content = "",
                promptTokens = 0,
                completionTokens = 0,
                errorMessage = e.message
            )
        }
    }

    fun destroy() {
        if (isInitialized) {
            isInitialized = false
            i( "VlmAnalyzer destroyed")
        }
    }

    companion object {
        private const val TAG = "VLM-VlmAnalyzer"

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }
}
