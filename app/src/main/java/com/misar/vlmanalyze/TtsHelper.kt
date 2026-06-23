// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

/**
 * Android Text-to-Speech helper using Java TextToSpeech API.
 * Simpler and more reliable than native OpenSL ES implementation.
 */
class TtsHelper(private val context: Context) {

    private var textToSpeech: TextToSpeech? = null
    private var isInitialized = false
    private var pendingText: String? = null

    interface Listener {
        fun onSpeakingComplete()
    }

    private var listener: Listener? = null

    // Debug log helper - only logs when debug mode is enabled
    private fun d(msg: String) {
        if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
    }

    // Info log helper - only logs when info mode is enabled
    private fun i(msg: String) {
        if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
    }

    // No native initialization needed - Android TTS is pure Java

    /**
     * Initialize TTS engine. Asynchronous - use Listener for confirmation.
     */
    fun initialize(listener: Listener?) {
        this.listener = listener

        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "TTS language not supported")
                    isInitialized = false
                    listener?.onSpeakingComplete()
                } else {
                    isInitialized = true
                    textToSpeech?.setSpeechRate(1.0f)  // Default rate
                    textToSpeech?.setPitch(1.0f)

                    // Set utterance progress listener
                    textToSpeech?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            d("TTS started")
                        }

                        override fun onDone(utteranceId: String?) {
                            d("TTS completed")
                            listener?.onSpeakingComplete()
                        }

                        override fun onError(utteranceId: String?) {
                            Log.e(TAG, "TTS error")
                            listener?.onSpeakingComplete()
                        }
                    })

                    i( "TTS initialized successfully")

                    // Speak pending text if any
                    pendingText?.let { speakInternal(it) }
                    pendingText = null
                }
            } else {
                Log.e(TAG, "TTS initialization failed")
                isInitialized = false
                listener?.onSpeakingComplete()
            }
        }
    }

    /**
     * Set speech rate (0.5 = slow, 1.0 = normal, 2.0 = fast)
     */
    fun setRate(rate: Float) {
        textToSpeech?.setSpeechRate(rate)
    }

    /**
     * Set pitch (0.5 = low, 1.0 = normal, 2.0 = high)
     */
    fun setPitch(pitch: Float) {
        textToSpeech?.setPitch(pitch)
    }

    /**
     * Speak text synchronously (blocks until done)
     */
    fun speak(text: String, rate: Float = 1.0f) {
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized, queuing: $text")
            pendingText = text
            return
        }

        setRate(rate)
        speakInternal(text)
    }

    private fun speakInternal(text: String) {
        // Strip markdown formatting
        val cleanText = stripMarkdown(text)

        if (cleanText.isEmpty()) {
            Log.w(TAG, "Empty text after stripping markdown")
            listener?.onSpeakingComplete()
            return
        }

        i( "Speaking: $cleanText")
        textToSpeech?.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, "utterance_${System.currentTimeMillis()}")
    }

    /**
     * Strip markdown formatting from text (matching C++ implementation)
     */
    private fun stripMarkdown(text: String): String {
        var result = text

        // Remove bold/italic markers
        result = result.replace(Regex("\\*+"), "")

        // Remove code blocks
        result = result.replace(Regex("```[\\s\\S]*?```"), "")

        // Remove inline code
        result = result.replace(Regex("`[^`]+`"), "")

        // Remove markdown links
        result = result.replace(Regex("\\[([^\\]]+)\\]\\([^)]+\\)"), "$1")

        // Remove markdown headers
        result = result.replace(Regex("^#{1,6}\\s+", RegexOption.MULTILINE), "")

        // Remove markdown lists
        result = result.replace(Regex("^\\s*[-*+]\\s+", RegexOption.MULTILINE), "")

        return result
    }

    /**
     * Shutdown TTS
     */
    fun shutdown() {
        if (textToSpeech != null) {
            textToSpeech?.stop()
            textToSpeech?.shutdown()
            textToSpeech = null
            isInitialized = false
        }
    }

    fun isReady(): Boolean = isInitialized

    companion object {
        private const val TAG = "VLM-TtsHelper"
    }
}
