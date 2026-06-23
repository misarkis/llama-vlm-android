// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.AudioManager
import android.media.AudioFocusRequest
import android.media.MediaRecorder
import android.os.Build
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.*
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import com.k2fsa.sherpa.onnx.*

/**
 * Voice input manager using sherpa-onnx for offline speech recognition.
 * Matches PC CPP pattern: continuous audio capture via callback, transcription on-demand.
 */
class VoiceInputManager(private val context: Context) {

    companion object {
        private const val TAG = "VLM-VoiceInputManager"
        private const val SAMPLE_RATE = 16000
        private const val POLL_INTERVAL_MS = 200L
        private const val TRANSCRIPTION_INTERVAL_MS = 500L
        private const val AUDIO_WINDOW_SECONDS = 5
        private const val MIN_AUDIO_FOR_TRANSCRIPTION = 1000

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
    }

    private var listener: VoiceInputListener? = null
    private var isListening = false
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // Audio focus for continuous recording
    private var audioManager: AudioManager? = null
    private var audioFocusRequest: Any? = null
    private var hadAudioFocus = false
    private val audioFocusListener = AudioManager.OnAudioFocusChangeListener {
        i( "Audio focus changed")
    }

    // Circular buffer for audio (stores last 10 seconds at 16kHz) - same as PC version
    private val audioBufferSize = 16000 * 10
    private val audioBuffer = ShortArray(audioBufferSize)
    private var bufferPosition = 0
    private var bufferFilled = 0

    // Audio read buffer (4096 bytes = 2048 samples at 16kHz = 128ms of audio)
    // Must use LITTLE_ENDIAN byte order for PCM audio
    private val readBuffer = java.nio.ByteBuffer.allocateDirect(4096).apply {
        order(java.nio.ByteOrder.LITTLE_ENDIAN)
    }

    // Recognizer (lazy initialization)
    private var recognizer: OfflineRecognizer? = null
    private val modelSubDir = "sherpa-onnx-whisper-base.en"
    private var modelDirPath: String? = null

    // VAD state
    private var lastSpeechEnergy = 0.0
    private var silenceFrames = 0
    private var speechDetected = false
    private val SPEECH_THRESHOLD = 50.0  // Energy threshold for speech detection
    private val SILENCE_FRAMES_REQUIRED = 3  // ~0.6 seconds of silence before transcription

    // Processing state (don't stop audio capture, just pause transcription)
    private var isProcessing = false  // Set true during VLM analysis

    fun setListener(listener: VoiceInputListener?) {
        this.listener = listener
    }

    private fun hasPermission(): Boolean {
        return android.content.pm.PackageManager.PERMISSION_GRANTED ==
            context.checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)
    }

    /**
     * Extract model files from assets to writable directory
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
                        d("Extracted: $sourceName (${destFile.length() / 1024 / 1024}MB)")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to extract $sourceName: ${e.message}")
                        return false
                    }
                } else {
                    d("Already exists: $sourceName")
                }
            }

            modelDirPath = modelDir.absolutePath
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup model files: ${e.message}", e)
            false
        }
    }

    private fun initializeRecognizer(): Boolean {
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

    fun startListening() {
        if (!hasPermission()) {
            listener?.onError("Microphone permission required")
            return
        }

        if (isListening) {
            Log.w(TAG, "Already listening")
            return
        }

        // Initialize recognizer (this extracts model files if needed)
        if (!initializeRecognizer()) {
            listener?.onError("Failed to initialize speech recognition")
            return
        }

        isListening = true
        bufferPosition = 0
        bufferFilled = 0
        listener?.onListeningStarted()
        i( "Starting voice listening with ${POLL_INTERVAL_MS}ms polling")

        // Request audio focus to keep microphone active
        requestAudioFocus()

        initAudioRecord()

        recordingJob = scope.launch {
            try {
                audioRecord?.startRecording()

                var pollCount = 0
                var speechDetected = false
                val startTime = System.currentTimeMillis()
                var lastTranscriptionTime = startTime  // Will wait 1.5s from this point
                var silentReads = 0
                while (isListening && isActive) {
                    try {
                        // Read audio from AudioRecord and update buffer
                        readBuffer.rewind()
                        val bytesRead = audioRecord?.read(readBuffer, readBuffer.remaining()) ?: 0

                        if (bytesRead > 0) {
                            val shortArray = ShortArray(bytesRead / 2)
                            readBuffer.asShortBuffer().get(shortArray)

                            // Detailed audio analysis
                            var sum = 0L
                            var maxSample = 0L
                            var zeroSamples = 0
                            for (s in shortArray) {
                                val abs = kotlin.math.abs(s.toLong())
                                sum += abs
                                if (abs > maxSample) maxSample = abs
                                if (s == 0.toShort()) zeroSamples++
                            }
                            val avg = if (shortArray.isNotEmpty()) sum / shortArray.size else 0
                            val zeroPercent = if (shortArray.isNotEmpty()) (zeroSamples * 100.0 / shortArray.size) else 0

                            // Log audio quality metrics (less frequently to avoid spam)
                            if (pollCount % 100 == 0) {
                                i( "Audio quality: avg=$avg, max=$maxSample, zeros=${zeroPercent.toInt()}%, bytesRead=$bytesRead, bufferFilled=$bufferFilled")
                            }
                            // Log when we have meaningful audio
                            if (maxSample > 500 && pollCount % 200 == 0) {
                                i( ">>> Significant audio detected: max=$maxSample, avg=$avg")
                            }

                            // Track silent reads
                            if (avg < 10 && maxSample < 50) {
                                silentReads++
                                if (silentReads > 20 && pollCount % 100 == 0) {
                                    Log.w(TAG, "WARNING: $silentReads consecutive silent reads - microphone may not be capturing!")
                                }
                            } else {
                                silentReads = 0
                            }

                            // UNPROCESSED source should not clip - use raw audio directly
                            updateAudioBuffer(shortArray)

                            // Calculate energy for VAD
                            val energy = calculateEnergy(shortArray)
                            val isSpeech = energy > SPEECH_THRESHOLD

                            if (isSpeech && !speechDetected) {
                                speechDetected = true
                                silenceFrames = 0
                                i( "Speech detected (energy=$energy, avg=$avg, max=$maxSample)")
                            } else if (isSpeech) {
                                silenceFrames = 0
                            } else if (speechDetected) {
                                silenceFrames++
                            }
                        } else {
                            // Handle read errors or no data
                            if (bytesRead < 0) {
                                Log.w(TAG, "AudioRecord read error: $bytesRead (error code)")
                            } else if (pollCount % 100 == 0) {
                                d("No audio data read (bytesRead=$bytesRead)")
                            }
                        }

                        // Periodic transcription every 500ms (like CPP version)
                        // But skip transcription if we're processing (VLM analysis in progress)
                        if (!isProcessing) {
                            val now = System.currentTimeMillis()
                            val elapsedSinceStart = now - startTime

                            // Wait at least 1 second before first transcription (buffer warmup)
                            if (elapsedSinceStart >= 1000) {
                                if (now - lastTranscriptionTime >= TRANSCRIPTION_INTERVAL_MS) {
                                    lastTranscriptionTime = now

                                    // Get ALL recent audio (like CPP get_recorded_audio)
                                    val recentAudio = getRecentAudio(AUDIO_WINDOW_SECONDS)
                                    if (recentAudio.isNotEmpty() && recentAudio.size >= SAMPLE_RATE * MIN_AUDIO_FOR_TRANSCRIPTION / 1000) {
                                        // Check audio quality before transcription (like CPP checks for empty buffer)
                                        var maxSample = 0L
                                        for (s in recentAudio) {
                                            val abs = kotlin.math.abs(s.toLong())
                                            if (abs > maxSample) maxSample = abs
                                        }

                                        if (maxSample >= 100) {  // Minimum audio level check (like CPP empty check)
                                            val text = transcribeAudio(recentAudio)
                                            // Clear buffer AFTER EVERY transcription (like CPP get_recorded_audio)
                                            clearAudioBuffer()

                                            if (text.isNotBlank()) {
                                                i( "Recognized: '$text' (maxSample=$maxSample)")
                                                listener?.onSpeechRecognized(text)
                                                val keyword = VoiceKeywords.matchKeyword(text)
                                                if (keyword != null) {
                                                    i( "Keyword detected: $keyword from '$text'")
                                                    listener?.onKeywordDetected(keyword)
                                                }
                                            }
                                        } else if (pollCount % 1000 == 0) {
                                            d("Audio too quiet (maxSample=$maxSample), skipping")
                                        }
                                    }
                                }
                            } else if (pollCount % 500 == 0) {
                                i( "Warming up buffer... ${elapsedSinceStart}ms / 1000ms")
                            }
                        } else {
                            // Still capturing audio, just not transcribing (buffer keeps rolling)
                            if (pollCount % 500 == 0) {
                                d("Skipping transcription - processing in progress (buffer still rolling)")
                            }
                        }

                        pollCount++
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in polling loop: ${e.message}", e)
                    }

                    delay(POLL_INTERVAL_MS)
                }
            } finally {
                audioRecord?.stop()
                audioRecord?.release()
                audioRecord = null
                isListening = false
                i( "Voice listening stopped")
            }
        }
    }

    private fun initAudioRecord() {
        // Use audio mode that keeps microphone active even when screen is off
        // This matches how third-party recording apps work

        // Use larger buffer size to prevent underruns
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            listener?.onError("Invalid audio buffer size: $bufferSize")
            return
        }

        i( "Min buffer size: $bufferSize bytes (${bufferSize / 2} samples)")

        // Try UNPROCESSED first (bypasses AGC/NS), fall back to MIC if it fails
        val audioSource = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
            i( "Using UNPROCESSED audio source (Android 7.0+)")
            MediaRecorder.AudioSource.UNPROCESSED
        } else {
            i( "Using MIC audio source (Android < 7.0)")
            MediaRecorder.AudioSource.MIC
        }

        try {
            audioRecord = AudioRecord(
                audioSource,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize * 4  // Use 4x minimum buffer for stability
            )

            if (audioRecord?.state == AudioRecord.STATE_INITIALIZED) {
                i( "AudioRecord initialized with source: $audioSource, buffer: ${bufferSize * 4}")

                // Try to disable audio effects for cleaner input
                try {
                    val sessionId = audioRecord?.audioSessionId ?: 0
                    if (sessionId > 0) {
                        if (android.media.audiofx.NoiseSuppressor.isAvailable()) {
                            val nss = android.media.audiofx.NoiseSuppressor.create(sessionId)
                            nss.enabled = false
                        }
                        if (android.media.audiofx.AutomaticGainControl.isAvailable()) {
                            val agc = android.media.audiofx.AutomaticGainControl.create(sessionId)
                            agc.enabled = false
                        }
                    }
                } catch (e: Exception) {
                    d("Could not disable audio effects: ${e.message}")
                }

                i( "AudioRecord ready - starting capture")
                return
            } else {
                Log.e(TAG, "AudioRecord initialization failed")
                audioRecord?.release()
                audioRecord = null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init AudioRecord: ${e.message}", e)
        }

        listener?.onError("Failed to initialize AudioRecord")
    }

    /**
     * Disable all audio effects to get raw, unprocessed audio
     */
    private fun disableAudioEffects() {
        val audioSessionId = audioRecord?.audioSessionId ?: return

        try {
            // Disable noise suppressor - let raw audio through
            if (android.media.audiofx.NoiseSuppressor.isAvailable()) {
                val noiseSuppressor = android.media.audiofx.NoiseSuppressor.create(audioSessionId)
                noiseSuppressor.enabled = false
                d("Noise suppressor disabled")
            }
        } catch (e: Exception) {
            d("Noise suppressor not available: ${e.message}")
        }

        try {
            // Disable acoustic echo canceller
            if (android.media.audiofx.AcousticEchoCanceler.isAvailable()) {
                val aec = android.media.audiofx.AcousticEchoCanceler.create(audioSessionId)
                aec.enabled = false
                d("AEC disabled")
            }
        } catch (e: Exception) {
            d("AEC not available: ${e.message}")
        }

        try {
            // Disable auto gain control - prevents over-amplification/clipping
            if (android.media.audiofx.AutomaticGainControl.isAvailable()) {
                val agc = android.media.audiofx.AutomaticGainControl.create(audioSessionId)
                agc.enabled = false
                d("AGC disabled")
            }
        } catch (e: Exception) {
            d("AGC not available: ${e.message}")
        }
    }

    private fun updateAudioBuffer(samples: ShortArray) {
        for (i in samples.indices) {
            val pos = (bufferPosition + i) % audioBufferSize
            audioBuffer[pos] = samples[i]
        }
        bufferPosition = (bufferPosition + samples.size) % audioBufferSize
        bufferFilled = minOf(bufferFilled + samples.size, audioBufferSize)

        // Debug: log first few samples to verify audio capture
        if (bufferFilled < 20) {
            val sampleValues = samples.take(5).joinToString(", ") { it.toString() }
            d("Audio samples captured: $sampleValues (bufferFilled=$bufferFilled)")
        }
    }

 
    private fun getRecentAudio(seconds: Int): ShortArray {
        val numSamples = SAMPLE_RATE * seconds
        val availableSamples = if (bufferFilled < audioBufferSize) bufferFilled else audioBufferSize
        if (availableSamples < numSamples) {
            return ShortArray(0)
        }

        val result = ShortArray(numSamples)
        for (i in 0 until numSamples) {
            val bufferPos = (bufferPosition - numSamples + i + audioBufferSize) % audioBufferSize
            result[i] = audioBuffer[bufferPos]
        }
        return result
    }

    // Clear the circular buffer after transcription (matches PC CPP pattern)
    private fun clearAudioBuffer() {
        bufferPosition = 0
        bufferFilled = 0
    }

    /**
     * Save current audio buffer to a WAV file for playback/verification
     * Returns the file path or null if save failed
     */
    fun saveAudioToWavFile(): String? {
        // Check if saving is enabled in config
        d("saveAudio config: ${Config.saveAudio}")
        if (!Config.saveAudio) {
            d("Audio saving disabled in config")
            return null
        }

        return try {
            val recentAudio = getRecentAudio(AUDIO_WINDOW_SECONDS)
            if (recentAudio.isEmpty()) {
                Log.w(TAG, "No audio data to save")
                return null
            }

            // Compute max/min for debug
            var maxSample = Short.MIN_VALUE
            var minSample = Short.MAX_VALUE
            for (s in recentAudio) {
                if (s > maxSample) maxSample = s
                if (s < minSample) minSample = s
            }
            i( "Audio range: $minSample to $maxSample")

            val timestamp = System.currentTimeMillis()
            val fileName = "VLM_AUDIO_${timestamp}.wav"

            // Save to public Downloads/VLM_Capture using MediaStore (Android 10+)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                val resolver = context.contentResolver
                val values = android.content.ContentValues().apply {
                    put(android.provider.MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(android.provider.MediaStore.Downloads.MIME_TYPE, "audio/wav")
                    put(android.provider.MediaStore.Downloads.RELATIVE_PATH, "${android.os.Environment.DIRECTORY_DOWNLOADS}/VLM_Capture")
                }

                val uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    resolver.openOutputStream(uri)?.use { outputStream ->
                        writeWavData(outputStream, recentAudio)
                        val path = "Downloads/VLM_Capture/$fileName"
                        i( "Saved audio to public Downloads: $path")
                        return path
                    }
                }
            }

            // Fallback: save to internal storage
            val capturesDir = File(context.filesDir, "audio_captures")
            if (!capturesDir.exists()) {
                capturesDir.mkdirs()
            }

            val outputFile = File(capturesDir, fileName)
            FileOutputStream(outputFile).use { fos ->
                writeWavData(fos, recentAudio)
            }

            val path = "Internal: ${outputFile.absolutePath}"
            i( "Saved audio to internal storage: $path")
            path
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save audio: ${e.message}", e)
            null
        }
    }

    /**
     * Write WAV file data to output stream (matching reference implementation)
     */
    private fun writeWavData(outputStream: java.io.OutputStream, recentAudio: ShortArray) {
        // Debug: log audio properties
        i( "Writing WAV: samples=${recentAudio.size}, sampleRate=$SAMPLE_RATE, duration=${recentAudio.size / SAMPLE_RATE}s")

        val totalAudioLen: Long = (recentAudio.size * 2).toLong()  // 2 bytes per sample
        val totalDataLen: Long = totalAudioLen + 36
        val byteRate: Long = (SAMPLE_RATE * 2).toLong()  // SampleRate * NumChannels * BitsPerSample / 8

        // Build 44-byte WAV header (exact match to reference)
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xffL).toByte()
        header[5] = ((totalDataLen shr 8) and 0xffL).toByte()
        header[6] = ((totalDataLen shr 16) and 0xffL).toByte()
        header[7] = ((totalDataLen shr 24) and 0xffL).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16.toByte()        // 4 bytes: size of 'fmt ' chunk
        header[17] = 0.toByte()
        header[18] = 0.toByte()
        header[19] = 0.toByte()
        header[20] = 1.toByte()        // format = 1 (PCM)
        header[21] = 0.toByte()
        header[22] = 1.toByte()        // NumChannels
        header[23] = 0.toByte()
        header[24] = (SAMPLE_RATE and 0xff).toByte()
        header[25] = ((SAMPLE_RATE shr 8) and 0xff).toByte()
        header[26] = ((SAMPLE_RATE shr 16) and 0xff).toByte()
        header[27] = ((SAMPLE_RATE shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xffL).toByte()
        header[29] = ((byteRate shr 8) and 0xffL).toByte()
        header[30] = ((byteRate shr 16) and 0xffL).toByte()
        header[31] = ((byteRate shr 24) and 0xffL).toByte()
        header[32] = 2.toByte()        // block align
        header[33] = 0.toByte()
        header[34] = 16.toByte()       // bits per sample
        header[35] = 0.toByte()
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xffL).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xffL).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xffL).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xffL).toByte()

        // Write header
        outputStream.write(header, 0, 44)

        // Write PCM audio data (convert ShortArray to ByteArray)
        val audioBytes = ByteArray(recentAudio.size * 2)
        var pos = 0
        for (sample in recentAudio) {
            audioBytes[pos++] = (sample.toInt() and 0xff).toByte()
            audioBytes[pos++] = ((sample.toInt() shr 8) and 0xff).toByte()
        }
        outputStream.write(audioBytes, 0, audioBytes.size)
        outputStream.flush()
    }

    // Calculate audio energy for VAD (Voice Activity Detection)
    private fun calculateEnergy(samples: ShortArray): Double {
        var sum = 0.0
        for (sample in samples) {
            val normalized = sample.toInt() / 32768.0
            sum += normalized * normalized
        }
        return sum / samples.size
    }

    private fun pcm16ToFloat(samples: ShortArray): FloatArray {
        return FloatArray(samples.size) {
            samples[it].toFloat() / 32768.0f
        }
    }

    private fun transcribeAudio(audio: ShortArray): String {
        try {
            val recognizer = recognizer ?: return ""

            // Verify audio data before transcription
            var maxSample = 0L
            for (s in audio) {
                val abs = kotlin.math.abs(s.toLong())
                if (abs > maxSample) maxSample = abs
            }
            if (maxSample < 10) {
                d("Skipping transcription: audio too quiet (max=$maxSample)")
                return ""
            }

            // Save audio if enabled (save BEFORE transcription to capture clean audio)
            if (Config.saveAudio) {
                val savedPath = saveAudioToWavFile()
                if (savedPath != null) {
                    i( "Audio saved to: $savedPath")
                }
            }

            // Verify audio data before transcription

            val floatAudio = pcm16ToFloat(audio)
            val stream = recognizer.createStream()
            stream.acceptWaveform(floatAudio, SAMPLE_RATE)
            recognizer.decode(stream)
            val result = recognizer.getResult(stream)

            if (result.text.isNotBlank()) {
                i( "Transcription: '${result.text}' (audio=${audio.size} samples, max=$maxSample)")
            }
            return result.text
        } catch (e: Exception) {
            Log.e(TAG, "sherpa-onnx error: ${e.message}", e)
            return ""
        }
    }

    fun stopListening() {
        isListening = false
        recordingJob?.cancel()
        recordingJob = null
        // Release audio focus when stopping
        releaseAudioFocus()
        i( "Voice listening cancelled")
    }

    /**
     * Request audio focus to keep microphone active during continuous recording
     */
    private fun requestAudioFocus() {
        try {
            audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()

                val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
                    .setAudioAttributes(audioAttributes)
                    .build()

                audioFocusRequest = request
                val result = audioManager?.requestAudioFocus(request)
                hadAudioFocus = result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
                i( "Audio focus request (O+): ${if (hadAudioFocus) "GRANTED" else "DENIED"}")
            } else {
                @Suppress("DEPRECATION")
                val result = audioManager?.requestAudioFocus(
                    audioFocusListener,
                    AudioManager.STREAM_VOICE_CALL,
                    AudioManager.AUDIOFOCUS_GAIN
                )
                hadAudioFocus = result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
                i( "Audio focus request (legacy): ${if (hadAudioFocus) "GRANTED" else "DENIED"}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to request audio focus: ${e.message}")
        }
    }

    /**
     * Release audio focus when stopping recording
     */
    private fun releaseAudioFocus() {
        try {
            if (audioFocusRequest != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                audioManager?.abandonAudioFocusRequest(audioFocusRequest as android.media.AudioFocusRequest)
            } else {
                @Suppress("DEPRECATION")
                audioManager?.abandonAudioFocus(null)
            }
            audioFocusRequest = null
            hadAudioFocus = false
            i( "Audio focus released")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to release audio focus: ${e.message}")
        }
    }

    /**
     * Set processing state - when true, transcription is paused but audio capture continues
     * Call this when VLM analysis starts, then call with false when analysis completes
     */
    fun setProcessing(processing: Boolean) {
        val oldState = isProcessing
        isProcessing = processing
        if (processing != oldState) {
            i( "Processing state: ${if (processing) "PAUSED (keeping audio buffer rolling)" else "RESUMED transcription"}")
        }
    }

    /**
     * Check if currently processing (transcription paused)
     */
    fun isCurrentlyProcessing(): Boolean = isProcessing
}
