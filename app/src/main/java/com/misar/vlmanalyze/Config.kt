// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Configuration loader for VLM-Analyze.
 * Loads from config.json in assets or storage.
 */
object Config {

    private const val TAG = "VLM-Config"

    // Info log helper - only logs when info mode is enabled
    private fun i(msg: String) {
        if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
    }

    // Default values
    private const val DEFAULT_MODEL = "unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL"
    private const val DEFAULT_API_URL = "http://192.168.1.46:8080/v1/chat/completions"
    private const val DEFAULT_SYSTEM_PROMPT = "You are a AI assistant. Please answer in details in English only. Just English. Do not use chineese. No chineese."
    private const val DEFAULT_USER_PROMPT = "What is on this screen right now? What should the person see? Tell them what matters. Keep it casual like you are explaining it to a friend."
    private const val DEFAULT_CAPTURE_COOLDOWN = 3
    private const val DEFAULT_MAX_TOKENS = 512
    private const val DEFAULT_TEMPERATURE = 0.9
    private const val DEFAULT_TOP_P = 0.9
    private const val DEFAULT_IMAGE_QUALITY = 100
    private const val DEFAULT_IMAGE_SCALE = 4
    private const val DEFAULT_IMAGE_SCALE_FACTOR_CAMERA = 0.4f
    private const val DEFAULT_IMAGE_SCALE_FACTOR_SCREEN = 0.25f
    private const val DEFAULT_TTS_RATE_ANALYSIS = 140
    private const val DEFAULT_TTS_RATE_RESULT = 180
    private const val DEFAULT_SAVE_AUDIO = false
    private const val DEFAULT_SAVE_SCREENSHOTS = false

    // Config values
    var model: String = DEFAULT_MODEL
    var apiUrl: String = DEFAULT_API_URL
    var systemPrompt: String = DEFAULT_SYSTEM_PROMPT
    var userPrompt: String = DEFAULT_USER_PROMPT
    var captureCooldown: Int = DEFAULT_CAPTURE_COOLDOWN
    var maxTokens: Int = DEFAULT_MAX_TOKENS
    var temperature: Double = DEFAULT_TEMPERATURE
    var topP: Double = DEFAULT_TOP_P
    var imageQuality: Int = DEFAULT_IMAGE_QUALITY
    var imageScale: Int = DEFAULT_IMAGE_SCALE
    var imageScaleFactorCamera: Float = DEFAULT_IMAGE_SCALE_FACTOR_CAMERA
    var imageScaleFactorScreen: Float = DEFAULT_IMAGE_SCALE_FACTOR_SCREEN
    var ttsRateAnalysis: Int = DEFAULT_TTS_RATE_ANALYSIS
    var ttsRateResult: Int = DEFAULT_TTS_RATE_RESULT
    var saveAudio: Boolean = DEFAULT_SAVE_AUDIO
    var saveScreenshots: Boolean = DEFAULT_SAVE_SCREENSHOTS

    /**
     * Load configuration from JSON file.
     */
    fun load(storagePath: String) {
        try {
            // Try to load from storage first
            val configFile = File(storagePath, "config.json")
            val jsonContent = if (configFile.exists()) {
                i( "Loading config from storage: $storagePath/config.json")
                configFile.readText()
            } else {
                i( "No config file found, using defaults")
                return
            }

            val json = JSONObject(jsonContent)

            model = json.optString("model", DEFAULT_MODEL)
            apiUrl = json.optString("api_url", DEFAULT_API_URL)
            systemPrompt = json.optString("system_prompt", DEFAULT_SYSTEM_PROMPT)
            userPrompt = json.optString("user_prompt", DEFAULT_USER_PROMPT)
            captureCooldown = json.optInt("capture_cooldown", DEFAULT_CAPTURE_COOLDOWN)
            maxTokens = json.optInt("max_tokens", DEFAULT_MAX_TOKENS)
            temperature = json.optDouble("temperature", DEFAULT_TEMPERATURE)
            topP = json.optDouble("top_p", DEFAULT_TOP_P)
            imageQuality = json.optInt("image_quality", DEFAULT_IMAGE_QUALITY)
            imageScale = json.optInt("image_scale", DEFAULT_IMAGE_SCALE)
            imageScaleFactorCamera = json.optString("image_scale_factor_camera", DEFAULT_IMAGE_SCALE_FACTOR_CAMERA.toString()).toFloatOrNull() ?: DEFAULT_IMAGE_SCALE_FACTOR_CAMERA
        imageScaleFactorScreen = json.optString("image_scale_factor_screen", DEFAULT_IMAGE_SCALE_FACTOR_SCREEN.toString()).toFloatOrNull() ?: DEFAULT_IMAGE_SCALE_FACTOR_SCREEN
            ttsRateAnalysis = json.optInt("tts_rate_analysis", DEFAULT_TTS_RATE_ANALYSIS)
            ttsRateResult = json.optInt("tts_rate_result", DEFAULT_TTS_RATE_RESULT)
            saveAudio = json.optBoolean("save_audio", DEFAULT_SAVE_AUDIO)
            saveScreenshots = json.optBoolean("save_screenshots", DEFAULT_SAVE_SCREENSHOTS)

            i( "Config loaded successfully")
            i( "  API URL: $apiUrl")
            i( "  Model: $model")
            i( "  Max tokens: $maxTokens")
            i( "  Save audio: $saveAudio")
            i( "  Save screenshots: $saveScreenshots")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load config, using defaults", e)
        }
    }

    /**
     * Save configuration to JSON file.
     */
    fun save(storagePath: String) {
        try {
            val file = File(storagePath, "config.json")
            val json = JSONObject().apply {
                put("model", model)
                put("api_url", apiUrl)
                put("system_prompt", systemPrompt)
                put("user_prompt", userPrompt)
                put("capture_cooldown", captureCooldown)
                put("max_tokens", maxTokens)
                put("temperature", temperature)
                put("top_p", topP)
                put("image_quality", imageQuality)
                put("image_scale", imageScale)
                put("image_scale_factor_camera", imageScaleFactorCamera)
            put("image_scale_factor_screen", imageScaleFactorScreen)
                put("tts_rate_analysis", ttsRateAnalysis)
                put("tts_rate_result", ttsRateResult)
                put("save_audio", saveAudio)
                put("save_screenshots", saveScreenshots)
            }
            file.writeText(json.toString(2))
            i( "Config saved to: $storagePath/config.json")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save config", e)
        }
    }

    /**
     * Reset to defaults.
     */
    fun reset() {
        model = DEFAULT_MODEL
        apiUrl = DEFAULT_API_URL
        systemPrompt = DEFAULT_SYSTEM_PROMPT
        userPrompt = DEFAULT_USER_PROMPT
        captureCooldown = DEFAULT_CAPTURE_COOLDOWN
        maxTokens = DEFAULT_MAX_TOKENS
        temperature = DEFAULT_TEMPERATURE
        topP = DEFAULT_TOP_P
        imageQuality = DEFAULT_IMAGE_QUALITY
        imageScale = DEFAULT_IMAGE_SCALE
        imageScaleFactorCamera = DEFAULT_IMAGE_SCALE_FACTOR_CAMERA
        imageScaleFactorScreen = DEFAULT_IMAGE_SCALE_FACTOR_SCREEN
        ttsRateAnalysis = DEFAULT_TTS_RATE_ANALYSIS
        ttsRateResult = DEFAULT_TTS_RATE_RESULT
        saveAudio = DEFAULT_SAVE_AUDIO
        saveScreenshots = DEFAULT_SAVE_SCREENSHOTS
        i( "Config reset to defaults")
    }
}
