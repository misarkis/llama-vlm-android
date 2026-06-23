// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

/**
 * Interface for voice input callbacks.
 */
interface VoiceInputListener {
    fun onListeningStarted()
    fun onSpeechRecognized(text: String)      // Partial results
    fun onFinalTranscript(text: String)       // Complete transcription
    fun onKeywordDetected(keyword: String)    // "ANALYZE" or "QUIT"
    fun onError(message: String)
}
