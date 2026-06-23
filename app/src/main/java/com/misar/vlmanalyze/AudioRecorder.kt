// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Android audio recorder for voice recognition.
 * Records at 16kHz mono (whisper.cpp requires this format).
 */
class AudioRecorder {

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var onAudioSample: ((ShortArray) -> Unit)? = null

    companion object {
        private const val TAG = "VLM-AudioRecorder"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }

    /**
     * Check if microphone permission is granted
     */
    fun hasPermission(context: android.content.Context): Boolean {
        return context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
    }

    /**
     * Start recording audio
     */
    @SuppressLint("MissingPermission")
    fun startRecording(context: android.content.Context, onSample: (ShortArray) -> Unit) {
        if (!hasPermission(context)) {
            Log.e(TAG, "Microphone permission not granted")
            return
        }

        this.onAudioSample = onSample

        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid buffer size")
            return
        }

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize * 2
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed")
                return
            }

            isRecording = true
            audioRecord?.startRecording()

            recordingThread = Thread {
                val buffer = ShortArray(bufferSize / 2)
                while (isRecording) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        val samples = ShortArray(read)
                        System.arraycopy(buffer, 0, samples, 0, read)
                        onAudioSample?.invoke(samples)
                    }
                }
            }.apply { start() }

            i( "Audio recording started")
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception: ${e.message}")
        }
    }

    /**
     * Stop recording audio
     */
    fun stopRecording() {
        isRecording = false
        recordingThread?.join()

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        i( "Audio recording stopped")
    }

    /**
     * Convert PCM16 samples to float (whisper.cpp format)
     */
    fun pcm16ToFloat(pcm16: ShortArray): FloatArray {
        return FloatArray(pcm16.size) {
            pcm16[it].toFloat() / 32768.0f
        }
    }

    /**
     * Convert PCM16 bytes to float
     */
    fun pcm16BytesToFloat(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val shortArray = ShortArray(bytes.size / 2)
        buffer.asShortBuffer()[shortArray]

        return pcm16ToFloat(shortArray)
    }
}
