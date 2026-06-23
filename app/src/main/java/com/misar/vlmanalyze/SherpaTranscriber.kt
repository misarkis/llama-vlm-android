// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.Context
import android.util.Log
import java.io.File
import com.k2fsa.sherpa.onnx.*

/**
 * Sherpa-onnx transcriber for the overlay service.
 * Transcribes recorded audio when Mic button is pressed again.
 */
class SherpaTranscriber(private val context: Context) {

    private var recognizer: OfflineRecognizer? = null
    private var modelDirPath: String? = null
    private val modelSubDir = "sherpa-onnx-whisper-base.en"

    companion object {
        private const val TAG = "VLM-SherpaTranscriber"
        private const val SAMPLE_RATE = 16000

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }

    /**
     * Initialize the recognizer (extracts model files from assets if needed)
     */
    fun initialize(): Boolean {
        return try {
            if (modelDirPath == null) {
                if (!setupModelFiles()) {
                    return false
                }
            }

            val encoderFile = File(modelDirPath!!, "base.en-encoder.int8.onnx")
            val decoderFile = File(modelDirPath!!, "base.en-decoder.int8.onnx")
            val tokensFile = File(modelDirPath!!, "base.en-tokens.txt")

            d("Model paths: encoder=${encoderFile.exists()}, decoder=${decoderFile.exists()}, tokens=${tokensFile.exists()}")

            if (!encoderFile.exists() || !decoderFile.exists() || !tokensFile.exists()) {
                Log.e(TAG, "Model files not found after extraction")
                return false
            }

            val featConfig = FeatureConfig(
                sampleRate = SAMPLE_RATE,
                featureDim = 80,
                dither = 0.0f
            )

            val whisperConfig = OfflineWhisperModelConfig(
                encoder = encoderFile.absolutePath,
                decoder = decoderFile.absolutePath,
                language = "en",
                task = "transcribe",
                tailPaddings = 1000
            )

            val modelConfig = OfflineModelConfig(
                whisper = whisperConfig,
                tokens = tokensFile.absolutePath,
                modelType = "whisper",
                numThreads = 2,
                debug = false,
                provider = "cpu"
            )

            val config = OfflineRecognizerConfig(
                featConfig = featConfig,
                modelConfig = modelConfig,
                decodingMethod = "greedy_search"
            )

            recognizer = OfflineRecognizer(null, config)
            i( "OfflineRecognizer initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize recognizer: ${e.message}", e)
            false
        }
    }

    /**
     * Extract model files from assets to files directory
     */
    private fun setupModelFiles(): Boolean {
        return try {
            val assetManager = context.applicationContext.assets
            val modelDir = File(context.applicationContext.filesDir, "sherpa-models")
            if (!modelDir.exists()) {
                modelDir.mkdirs()
            }

            val filesToExtract = listOf(
                "base.en-encoder.int8.onnx" to File(modelDir, "base.en-encoder.int8.onnx"),
                "base.en-decoder.int8.onnx" to File(modelDir, "base.en-decoder.int8.onnx"),
                "base.en-tokens.txt" to File(modelDir, "base.en-tokens.txt")
            )

            for ((sourceName, destFile) in filesToExtract) {
                if (!destFile.exists()) {
                    val sourcePath = "$modelSubDir/$sourceName"
                    try {
                        assetManager.open(sourcePath).use { input ->
                            destFile.outputStream().use { output ->
                                input.copyTo(output)
                            }
                        }
                        d("Extracted: $sourceName")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to extract $sourceName: ${e.message}")
                        return false
                    }
                }
            }

            modelDirPath = modelDir.absolutePath
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup model files: ${e.message}", e)
            false
        }
    }

    /**
     * Transcribe audio samples to text
     * @param samples PCM16 audio samples at 16kHz
     * @return Transcribed text
     */
    fun transcribe(samples: List<Short>): String {
        try {
            val recognizer = recognizer ?: return ""

            if (samples.isEmpty()) {
                d("No audio samples to transcribe")
                return ""
            }

            // Verify audio quality
            var maxSample = 0L
            for (s in samples) {
                val abs = kotlin.math.abs(s.toLong())
                if (abs > maxSample) maxSample = abs
            }
            if (maxSample < 10) {
                d("Skipping transcription: audio too quiet (max=$maxSample)")
                return ""
            }

            i( "Transcribing ${samples.size} samples (max=$maxSample)")

            // Convert PCM16 to float
            val floatAudio = pcm16ToFloat(samples.toShortArray())

            // Create stream and process
            val stream = recognizer.createStream()
            stream.acceptWaveform(floatAudio, SAMPLE_RATE)
            recognizer.decode(stream)
            val result = recognizer.getResult(stream)

            i( "Transcription result: '${result.text}'")
            return result.text
        } catch (e: Exception) {
            Log.e(TAG, "Transcription error: ${e.message}", e)
            return ""
        }
    }

    /**
     * Convert PCM16 short array to float array
     */
    private fun pcm16ToFloat(pcm16: ShortArray): FloatArray {
        return FloatArray(pcm16.size) {
            pcm16[it].toFloat() / 32768.0f
        }
    }
}
