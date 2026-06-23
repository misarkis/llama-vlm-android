// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

/**
 * Keyword matching for voice commands.
 */
object VoiceKeywords {
    val ANALYZE_PHRASES = listOf(
        "analyze",
        "analyze my screen",
        "analyze this",
        "analyze this screen",
        "analyze this image",
        "what is on my screen",
        "what is on the screen",
        "what do you see",
        "what do you see on my screen",
        "what do you see in this image",
        "tell me what is on the screen",
        "tell me what you see",
        "explain what you see",
        "explain this",
        "explain what do you see",
        "describe this",
        "describe this screen",
        "describe what you see",
        "tell me about this",
        "tell me about the screen",
        "what do you see on this screen"
    )

    val QUIT_PHRASES = listOf(
        "quit",
        "exit",
        "stop",
        "close"
    )

    fun matchKeyword(text: String): String? {
        val normalized = text.lowercase().trim()

        // Exact match first
        if (normalized in ANALYZE_PHRASES) return "ANALYZE"
        if (normalized in QUIT_PHRASES) return "QUIT"

        // Partial match (contains key phrase)
        for (phrase in ANALYZE_PHRASES) {
            if (normalized.contains(phrase) || phrase.contains(normalized)) {
                return "ANALYZE"
            }
        }

        for (phrase in QUIT_PHRASES) {
            if (normalized.contains(phrase) || phrase.contains(normalized)) {
                return "QUIT"
            }
        }

        return null
    }
}
