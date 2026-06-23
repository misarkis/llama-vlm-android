// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.TimeUnit

/**
 * Android HTTP client for VLM API calls.
 * Simple implementation without SSE streaming dependency.
 */
class HttpClient private constructor() {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(0, TimeUnit.SECONDS)
        .build()

    private val mediaType = "application/json; charset=utf-8".toMediaType()

    data class ApiResult(
        val success: Boolean,
        val content: String,
        val promptTokens: Int,
        val completionTokens: Int,
        val totalTimeMs: Long = 0,
        val errorMessage: String? = null
    )

    interface StreamCallback {
        fun onChunk(chunk: String, totalTokens: Int)
    }

    /**
     * Send chat completion request with streaming
     */
    suspend fun chatCompletion(
        model: String,
        messages: List<Map<String, Any>>,
        maxTokens: Int = 1024,
        temperature: Double = 0.9,
        topP: Double = 0.9,
        streamCallback: StreamCallback? = null
    ): ApiResult = withContext(Dispatchers.IO) {
        val payload = JSONObject().apply {
            put("model", model)
            put("stream", true)
            put("max_tokens", maxTokens)
            put("temperature", temperature)
            put("top_p", topP)
            put("messages", messages.toJSONArray())
        }

        val request = Request.Builder()
            .url(Config.apiUrl)
            .post(payload.toString().toRequestBody(mediaType))
            .addHeader("Content-Type", "application/json")
            .build()

        var promptTokens = 0
        var completionTokens = 0
        val accumulatedContent = StringBuilder()
        var error: String? = null
        val startTime = System.currentTimeMillis()

        try {
            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                error = "HTTP error: ${response.code}"
                Log.e(TAG, "HTTP error: ${response.code}")
                return@withContext ApiResult(
                    success = false,
                    content = "",
                    promptTokens = 0,
                    completionTokens = 0,
                    errorMessage = error
                )
            }

            val body = response.body?.string() ?: ""
            i( "Response received: ${body.length} bytes")

            // Parse SSE stream manually
            val lines = body.split("\n", "\r\n")
            for (line in lines) {
                if (line.startsWith("data: ")) {
                    val data = line.substringAfter("data: ").trim()
                    if (data == "[DONE]" || data.isEmpty()) continue

                    try {
                        val json = JSONObject(data)

                        // Log usage info when found
                        if (json.has("usage")) {
                            val usage = json.getJSONObject("usage")
                            promptTokens = usage.getInt("prompt_tokens")
                            completionTokens = usage.getInt("completion_tokens")
                            i( "Usage found: prompt=$promptTokens, completion=$completionTokens")
                        } else if (json.has("choices") && json.getJSONArray("choices").length() > 0) {
                            val choice = json.getJSONArray("choices").getJSONObject(0)
                            if (choice.has("usage")) {
                                val usage = choice.getJSONObject("usage")
                                promptTokens = usage.getInt("prompt_tokens")
                                completionTokens = usage.getInt("completion_tokens")
                                i( "Usage in choices: prompt=$promptTokens, completion=$completionTokens")
                            }
                        }

                        val content = if (json.has("choices") && json.getJSONArray("choices").length() > 0) {
                            val choice = json.getJSONArray("choices").getJSONObject(0)
                            val delta = if (choice.has("delta")) {
                                choice.getJSONObject("delta")
                            } else if (choice.has("message")) {
                                choice.getJSONObject("message")
                            } else {
                                JSONObject()
                            }
                            val rawContent = delta.optString("content", null)
                            rawContent ?: ""
                        } else {
                            ""
                        }

                        if (content.isNotEmpty() && content != "null") {
                            accumulatedContent.append(content)
                            val totalTokens = if (promptTokens > 0 || completionTokens > 0) {
                                promptTokens + completionTokens
                            } else {
                                accumulatedContent.length / 4  // Estimate: 1 token ~= 4 chars
                            }
                            streamCallback?.onChunk(content, totalTokens)
                        }
                    } catch (e: Exception) {
                        // Skip invalid JSON lines
                    }
                }
            }

        } catch (e: Exception) {
            error = "Request failed: ${e.message}"
            Log.e(TAG, "Request error", e)
        }

        val totalTimeMs = System.currentTimeMillis() - startTime

        // Fallback: estimate tokens if API didn't provide usage (1 token ~= 4 chars for English)
        val finalPromptTokens = if (promptTokens > 0) promptTokens else accumulatedContent.length / 4
        val finalCompletionTokens = if (completionTokens > 0) completionTokens else accumulatedContent.length / 6

        i( "Final tokens: prompt=$finalPromptTokens, completion=$finalCompletionTokens, total=${finalPromptTokens + finalCompletionTokens}")
        ApiResult(
            success = error == null && accumulatedContent.isNotEmpty(),
            content = accumulatedContent.toString(),
            promptTokens = finalPromptTokens,
            completionTokens = finalCompletionTokens,
            totalTimeMs = totalTimeMs,
            errorMessage = error
        )
    }

    private fun List<Map<String, Any>>.toJSONArray(): org.json.JSONArray {
        val array = org.json.JSONArray()
        for (msg in this) {
            val obj = org.json.JSONObject()
            for ((key, value) in msg) {
                when (value) {
                    is String -> obj.put(key, value)
                    is Int -> obj.put(key, value)
                    is Long -> obj.put(key, value)
                    is Double -> obj.put(key, value)
                    is Boolean -> obj.put(key, value)
                    is List<*> -> obj.put(key, value.toListJSONArray())
                    is Map<*, *> -> obj.put(key, value.toJSONObject())
                    else -> obj.put(key, value?.toString() ?: JSONObject.NULL)
                }
            }
            array.put(obj)
        }
        return array
    }

    private fun List<*>.toListJSONArray(): org.json.JSONArray {
        val array = org.json.JSONArray()
        for (item in this) {
            when (item) {
                is String -> array.put(item)
                is Int -> array.put(item)
                is Long -> array.put(item)
                is Double -> array.put(item)
                is Boolean -> array.put(item)
                is List<*> -> array.put(item.toListJSONArray())
                is Map<*, *> -> array.put(item.toJSONObject())
                else -> array.put(item?.toString() ?: JSONObject.NULL)
            }
        }
        return array
    }

    private fun Map<*, *>.toJSONObject(): org.json.JSONObject {
        val obj = org.json.JSONObject()
        for ((key, value) in this) {
            if (key is String) {
                when (value) {
                    is String -> obj.put(key, value)
                    is Int -> obj.put(key, value)
                    is Long -> obj.put(key, value)
                    is Double -> obj.put(key, value)
                    is Boolean -> obj.put(key, value)
                    is List<*> -> obj.put(key, value.toListJSONArray())
                    is Map<*, *> -> obj.put(key, value.toJSONObject())
                    else -> obj.put(key, value?.toString() ?: JSONObject.NULL)
                }
            }
        }
        return obj
    }

    companion object {
        private const val TAG = "VLM-HttpClient"
        val instance = HttpClient()

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }
}
