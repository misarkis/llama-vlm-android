// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.core.content.ContextCompat
import android.util.Log
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * LocalBackend - Connects to locally running llama-server via HTTP.
 *
 * The LlamaServerService runs llama-server as a foreground service on localhost:8080.
 * This backend sends HTTP requests to the local server for VLM inference.
 *
 * Architecture:
 * - LlamaServerService extracts binary from APK assets and starts llama-server process
 * - llama-server binds to 127.0.0.1:8080
 * - LocalBackend sends HTTP POST requests to /v1/chat/completions
 * - Response is streamed back as SSE (Server-Sent Events)
 */
class LocalBackend(private val context: Context) : InferenceBackend {

    private val broadcastReceiver = LocalBroadcastReceiver()
    private var serverStatus = ServerStatus.STOPPED

    enum class ServerStatus {
        STOPPED,
        STARTING,
        RUNNING,
        ERROR
    }

    private var currentConfig: InferenceBackend.BackendConfig? = null

    companion object {
        private const val TAG = "VLM-LocalBackend"
        private const val SERVER_HOST = "127.0.0.1"
        private const val SERVER_PORT = 8080
        private const val SERVER_BASE_URL = "http://127.0.0.1:8080"

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }

    override fun initialize(config: InferenceBackend.BackendConfig): Boolean {
        i( "Initializing LocalBackend with model: ${config.modelPath}")
        currentConfig = config

        // Register broadcast receiver for server status updates
        val filter = IntentFilter().apply {
            addAction(LlamaServerService.ACTION_SERVER_STARTED)
            addAction(LlamaServerService.ACTION_SERVER_STOPPED)
            addAction(LlamaServerService.ACTION_SERVER_ERROR)
        }
        ContextCompat.registerReceiver(
            context,
            broadcastReceiver,
            filter,
            Context.RECEIVER_NOT_EXPORTED
        )

        return true
    }

    /**
     * Start the local server service.
     * This triggers LlamaServerService to extract the binary and start llama-server.
     */
    fun startServer(
        port: Int = 8080,
        nCtx: Int = 4096,
        nBatch: Int = 512,
        verbose: Boolean = false,
        htpSessions: Int = 1
    ) {
        val config = currentConfig ?: run {
            Log.e(TAG, "Cannot start server: not initialized")
            return
        }

        i( "Starting local server: model=${config.modelPath}, backend=${config.backend}, port=$port, ctx=$nCtx, batch=$nBatch, htpSessions=$htpSessions")
        serverStatus = ServerStatus.STARTING

        LlamaServerService.startService(
            context = context,
            modelPath = config.modelPath,
            mmprojPath = config.mmprojPath,
            backend = config.backend,
            port = port,
            nCtx = nCtx,
            nBatch = nBatch,
            verbose = verbose,
            htpSessions = htpSessions
        )
    }

    /**
     * Stop the local server service.
     * This triggers LlamaServerService to stop llama-server and free resources.
     */
    fun stopServer() {
        i( "Stopping local server")
        LlamaServerService.stopService(context)
        serverStatus = ServerStatus.STOPPED
    }

    /**
     * Check if the local server is running and ready.
     */
    fun isServerReady(): Boolean {
        return serverStatus == ServerStatus.RUNNING && LlamaServerService.isServerReady()
    }

    /**
     * Get current server status.
     */
    fun getServerStatus(): ServerStatus = serverStatus

    override suspend fun analyze(imageBytes: ByteArray, prompt: String, systemPrompt: String?): InferenceBackend.InferenceResult {
        // Check if server is ready
        if (!isServerReady()) {
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Local server is not running. Call startServer() first.",
                timings = null
            )
        }

        try {
            // Send request to local llama-server
            val result = sendInferenceRequest(imageBytes, prompt)
            return result
        } catch (e: Exception) {
            Log.e(TAG, "Inference request failed", e)
            return InferenceBackend.InferenceResult(
                success = false,
                content = "Inference failed: ${e.message}",
                timings = null
            )
        }
    }

    /**
     * Send inference request to local llama-server.
     * Uses the OpenAI-compatible API format.
     */
    private fun sendInferenceRequest(imageBytes: ByteArray, prompt: String): InferenceBackend.InferenceResult {
        val config = currentConfig ?: throw IllegalStateException("Not initialized")

        // Encode image to base64
        val base64Image = android.util.Base64.encodeToString(imageBytes, android.util.Base64.DEFAULT)

        // Build JSON request
        val requestBody = buildJsonRequest(base64Image, prompt, config)

        d("Sending request to $SERVER_BASE_URL/v1/chat/completions")

        val url = URL("$SERVER_BASE_URL/v1/chat/completions")
        val conn = url.openConnection() as HttpURLConnection

        try {
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.connectTimeout = 0  // No timeout for long generations
            conn.readTimeout = 0
            conn.doOutput = true

            // Send request body
            conn.outputStream.use { os ->
                os.write(requestBody.toByteArray())
                os.flush()
            }

            val responseCode = conn.responseCode
            val inputStream = if (responseCode >= 400) conn.errorStream else conn.inputStream

            val response = inputStream?.bufferedReader()?.use { it.readText() } ?: ""
            conn.disconnect()

            if (responseCode >= 400) {
                Log.e(TAG, "Server error: $responseCode - $response")
                return InferenceBackend.InferenceResult(
                    success = false,
                    content = "Server returned error $responseCode",
                    timings = null
                )
            }

            // Parse response
            val content = extractContentFromResponse(response)
            val timings = extractTimingsFromResponse(response)

            i( "Inference complete: ${content.length} chars")

            return InferenceBackend.InferenceResult(
                success = true,
                content = content,
                timings = timings
            )

        } catch (e: Exception) {
            Log.e(TAG, "Request failed", e)
            throw e
        } finally {
            conn.disconnect()
        }
    }

    /**
     * Build JSON request for llama-server.
     */
    private fun buildJsonRequest(base64Image: String, prompt: String, config: InferenceBackend.BackendConfig): String {
        // Escape special characters in prompt
        val escapedPrompt = prompt.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")

        // Build JSON manually (no external library)
        return """
        {
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "image_url",
                  "image_url": {
                    "url": "data:image/jpeg;base64,$base64Image"
                  }
                },
                {
                  "type": "text",
                  "text": "$escapedPrompt"
                }
              ]
            }
          ],
          "stream": false,
          "max_tokens": ${config.maxTokens},
          "temperature": ${config.temperature},
          "top_p": ${config.topP}
        }
        """.trimIndent()
    }

    /**
     * Extract content from JSON response.
     */
    private fun extractContentFromResponse(response: String): String {
        // Simple JSON parsing - extract content from choices[0].message.content
        return try {
            val contentStart = response.indexOf("\"content\"")
            if (contentStart == -1) return response

            val quoteStart = response.indexOf('"', contentStart + 9)
            val quoteEnd = response.indexOf('"', quoteStart + 1)

            if (quoteStart == -1 || quoteEnd == -1) return response

            // Handle escaped quotes
            var content = response.substring(quoteStart + 1, quoteEnd)
            content = content.replace("\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")

            content
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse response", e)
            response
        }
    }

    /**
     * Extract timing information from JSON response.
     */
    private fun extractTimingsFromResponse(response: String): InferenceBackend.InferenceResult.Timings? {
        return try {
            // Look for prompt_tokens and completion_tokens in usage
            val promptTokens = extractInt(response, "prompt_tokens")
            val completionTokens = extractInt(response, "completion_tokens")

            // Look for timing info in timings object
            val promptPerSecond = extractDouble(response, "prompt_per_second")
            val predictedPerSecond = extractDouble(response, "predicted_per_second")

            InferenceBackend.InferenceResult.Timings(
                promptTokens = promptTokens,
                completionTokens = completionTokens,
                promptPerSecond = if (promptPerSecond > 0) promptPerSecond else null,
                predictedPerSecond = if (predictedPerSecond > 0) predictedPerSecond else null
            )
        } catch (e: Exception) {
            null
        }
    }

    private fun extractInt(json: String, key: String): Int {
        val pattern = "\"$key\"\\s*:\\s*(\\d+)"
        val regex = Regex(pattern)
        return regex.find(json)?.groupValues?.get(1)?.toIntOrNull() ?: 0
    }

    private fun extractDouble(json: String, key: String): Double {
        val pattern = "\"$key\"\\s*:\\s*([\\d.]+)"
        val regex = Regex(pattern)
        return regex.find(json)?.groupValues?.get(1)?.toDoubleOrNull() ?: 0.0
    }

    override suspend fun analyzeRgbBytes(rgbBytes: ByteArray, width: Int, height: Int, prompt: String, systemPrompt: String?): InferenceBackend.InferenceResult {
        // For LocalBackend (HTTP server mode), we still need to convert RGB to JPEG
        // since the server expects JPEG input. This is a fallback for compatibility.
        try {
            d("RGB bytes mode - converting to JPEG for server")

            // Create bitmap from RGB bytes
            val pixels = IntArray(width * height)
            for (i in rgbBytes.indices step 3) {
                val pixelIndex = i / 3
                val r = rgbBytes[i].toInt() and 0xFF
                val g = rgbBytes[i + 1].toInt() and 0xFF
                val b = rgbBytes[i + 2].toInt() and 0xFF
                pixels[pixelIndex] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }

            val bitmap = android.graphics.Bitmap.createBitmap(pixels, width, height, android.graphics.Bitmap.Config.ARGB_8888)
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, outputStream)
            bitmap.recycle()

            val imageBytes = outputStream.toByteArray()
            return analyze(imageBytes, prompt, systemPrompt)
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
        i( "Destroying LocalBackend")
        try {
            context.unregisterReceiver(broadcastReceiver)
        } catch (e: Exception) {
            Log.w(TAG, "Receiver already unregistered", e)
        }
        stopServer()
    }

    /**
     * Broadcast receiver for server status updates.
     */
    private inner class LocalBroadcastReceiver : android.content.BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                LlamaServerService.ACTION_SERVER_STARTED -> {
                    i( "Server started")
                    serverStatus = ServerStatus.RUNNING
                }
                LlamaServerService.ACTION_SERVER_STOPPED -> {
                    i( "Server stopped")
                    serverStatus = ServerStatus.STOPPED
                }
                LlamaServerService.ACTION_SERVER_ERROR -> {
                    val error = intent.getStringExtra(LlamaServerService.EXTRA_ERROR_MESSAGE)
                    Log.e(TAG, "Server error: $error")
                    serverStatus = ServerStatus.ERROR
                }
            }
        }
    }
}
