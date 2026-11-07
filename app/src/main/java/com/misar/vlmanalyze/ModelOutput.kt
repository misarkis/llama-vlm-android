// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

object ModelOutput {
    private const val THINK_START = "<think>"
    private const val THINK_END = "</think>"
    private const val NO_ANSWER = "The model ran out of tokens while thinking, so there is no answer."

    /**
     * Returns the answer without the model's thinking block.
     * Text from models that do not think is returned unchanged.
     */
    fun stripThinking(text: String): String {
        val end = text.indexOf(THINK_END)
        if (end != -1) {
            return text.substring(end + THINK_END.length).trim()
        }
        return if (text.trimStart().startsWith(THINK_START)) NO_ANSWER else text
    }
}
