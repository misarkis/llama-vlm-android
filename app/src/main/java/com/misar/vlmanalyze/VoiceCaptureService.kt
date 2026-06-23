// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.*
import com.k2fsa.sherpa.onnx.*
import java.io.File

/**
 * Foreground service for continuous voice capture.
 * Keeps microphone active even when app is in background.
 */
class VoiceCaptureService : Service() {

    private var audioManager: AudioManager? = null
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val circularBuffer = ShortArray(16000 * 10) // 10 seconds at 16kHz
    private var bufferPosition = 0
    private var bufferFilled = 0
    private val readBuffer = java.nio.ByteBuffer.allocateDirect(4096).apply {
        order(java.nio.ByteOrder.LITTLE_ENDIAN)
    }
    private var recognizer: OfflineRecognizer? = null
    private var lastTranscriptionTime = 0L

    companion object {
        private const val TAG = "VLM-VoiceCaptureSvc"
        private const val NOTIFICATION_ID = 3
        private const val CHANNEL_ID = "vlm_voice_channel"
        private const val TRANSCRIPTION_INTERVAL_MS = 500L

        const val ACTION_START = "com.misar.vlmanalyze.VOICE_START"
        const val ACTION_STOP = "com.misar.vlmanalyze.VOICE_STOP"

        const val BROADCAST_KEYWORD = "com.misar.vlmanalyze.KEYWORD"
        const val EXTRA_KEYWORD = "keyword"

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        fun startService(context: Context) {
            i( "startService called")
            val intent = Intent(context, VoiceCaptureService::class.java).apply {
                action = ACTION_START
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stopService(context: Context) {
            i( "stopService called")
            context.stopService(Intent(context, VoiceCaptureService::class.java).apply {
                action = ACTION_STOP
            })
        }
    }

    override fun onCreate() {
        super.onCreate()
        i( "Service created")
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        i( "onStartCommand: ${intent?.action}")

        when (intent?.action) {
            ACTION_START -> {
                val notification = createNotification("🎤 Listening...")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
                } else {
                    startForeground(NOTIFICATION_ID, notification)
                }
                startRecording()
            }
            ACTION_STOP -> {
                stopRecording()
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        i( "Service destroying")
        stopRecording()
        scope.cancel()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Voice Capture",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun createNotification(status: String): Notification {
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VLM Voice Capture")
            .setContentText(status)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
    }

    private fun setupModelFiles(): Boolean {
        return try {
            val modelDir = File(filesDir, "sherpa-models")
            if (!modelDir.exists()) {
                modelDir.mkdirs()
            }

            val encoderFile = File(modelDir, "base.en-encoder.int8.onnx")
            val decoderFile = File(modelDir, "base.en-decoder.int8.onnx")
            val tokensFile = File(modelDir, "base.en-tokens.txt")

            if (!encoderFile.exists() || !decoderFile.exists() || !tokensFile.exists()) {
                d("Copying model files from assets")
                applicationContext.assets.open("base.en-encoder.int8.onnx").use { input ->
                    encoderFile.outputStream().use { output -> input.copyTo(output) }
                }
                applicationContext.assets.open("base.en-decoder.int8.onnx").use { input ->
                    decoderFile.outputStream().use { output -> input.copyTo(output) }
                }
                applicationContext.assets.open("base.en-tokens.txt").use { input ->
                    tokensFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup model files: ${e.message}", e)
            false
        }
    }

    private fun initializeRecognizer(): Boolean {
        return try {
            if (!setupModelFiles()) {
                return false
            }

            val modelDir = File(filesDir, "sherpa-models")
            val encoderFile = File(modelDir, "base.en-encoder.int8.onnx").absolutePath
            val decoderFile = File(modelDir, "base.en-decoder.int8.onnx").absolutePath
            val tokensFile = File(modelDir, "base.en-tokens.txt").absolutePath

            val whisperConfig = OfflineWhisperModelConfig(
                encoder = encoderFile,
                decoder = decoderFile,
                language = "en",
                task = "transcribe",
                tailPaddings = 1000
            )

            val modelConfig = OfflineModelConfig(
                whisper = whisperConfig,
                tokens = tokensFile,
                modelType = "whisper",
                numThreads = 2,
                debug = false,
                provider = "cpu"
            )

            val featConfig = FeatureConfig(
                sampleRate = 16000,
                featureDim = 80,
                dither = 0.0f
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

    private fun startRecording() {
        if (isRecording) {
            Log.w(TAG, "Already recording")
            return
        }

        // Initialize recognizer first
        if (!initializeRecognizer()) {
            Log.e(TAG, "Failed to initialize recognizer")
            return
        }

        try {
            // Request audio focus
            audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            val audioAttributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val request = android.media.AudioFocusRequest.Builder(
                    AudioManager.AUDIOFOCUS_GAIN
                ).setAudioAttributes(audioAttributes).build()
                audioManager?.requestAudioFocus(request)
            } else {
                @Suppress("DEPRECATION")
                audioManager?.requestAudioFocus(
                    null,
                    AudioManager.STREAM_VOICE_CALL,
                    AudioManager.AUDIOFOCUS_GAIN
                )
            }

            // Initialize AudioRecord
            val bufferSize = AudioRecord.getMinBufferSize(
                16000,
                android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_16BIT
            )
            i( "Min buffer size: $bufferSize bytes")

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.UNPROCESSED,
                16000,
                android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_16BIT,
                bufferSize * 4
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                return
            }

            audioRecord?.startRecording()
            isRecording = true
            lastTranscriptionTime = System.currentTimeMillis()
            i( "Recording started")

            // Start recording thread
            scope.launch {
                var pollCount = 0
                val startTime = System.currentTimeMillis()
                while (isRecording && isActive) {
                    try {
                        readBuffer.rewind()
                        val bytesRead = audioRecord?.read(readBuffer, readBuffer.remaining()) ?: 0

                        if (bytesRead > 0) {
                            val shortArray = ShortArray(bytesRead / 2)
                            readBuffer.asShortBuffer().get(shortArray)
                            writeSamplesToBuffer(shortArray)

                            // Transcribe every 500ms after warmup
                            val now = System.currentTimeMillis()
                            if (now - startTime >= 1000 && now - lastTranscriptionTime >= TRANSCRIPTION_INTERVAL_MS) {
                                lastTranscriptionTime = now
                                doTranscribe()
                            }
                        }

                        // Log audio quality periodically
                        if (pollCount % 500 == 0) {
                            logAudioQuality()
                        }
                        pollCount++

                    } catch (e: Exception) {
                        Log.e(TAG, "Error in recording loop: ${e.message}", e)
                    }
                    delay(200)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
        }
    }

    private fun writeSamplesToBuffer(samples: ShortArray) {
        for (sample in samples) {
            circularBuffer[bufferPosition] = sample
            bufferPosition = (bufferPosition + 1) % circularBuffer.size
            if (bufferFilled < circularBuffer.size) {
                bufferFilled++
            }
        }
    }

    private fun logAudioQuality() {
        val samplesToCheck = kotlin.math.min(bufferFilled, 1000)
        if (samplesToCheck == 0) return

        var sum = 0L
        var maxSample = 0L
        var zeroSamples = 0
        val startPos = if (bufferFilled >= circularBuffer.size) bufferPosition else 0

        for (i in 0 until samplesToCheck) {
            val s = circularBuffer[(startPos + i) % circularBuffer.size].toLong()
            val abs = kotlin.math.abs(s)
            sum += abs
            if (abs > maxSample) maxSample = abs
            if (s == 0L) zeroSamples++
        }

        val avg = sum / samplesToCheck
        val zeroPercent = (zeroSamples * 100.0 / samplesToCheck)

        i( "Audio quality: avg=$avg, max=$maxSample, zeros=${zeroPercent.toInt()}%, bufferFilled=$bufferFilled")
    }

    private fun doTranscribe() {
        val recentAudio = getRecentAudio(5) // 5 seconds
        if (recentAudio.isEmpty() || recentAudio.size < 1600) return

        var maxSample = 0L
        for (s in recentAudio) {
            val abs = kotlin.math.abs(s.toLong())
            if (abs > maxSample) maxSample = abs
        }

        if (maxSample < 100) return // Skip if too quiet

        val text = recognizeAudio(recentAudio)
        if (text.isNotBlank()) {
            i( "Transcript: '$text' (maxSample=$maxSample)")

            val keyword = VoiceKeywords.matchKeyword(text)
            if (keyword != null) {
                i( "Keyword detected: $keyword")
                broadcastKeyword(keyword)
            }
        }
    }

    private fun getRecentAudio(seconds: Int): ShortArray {
        val size = kotlin.math.min(bufferFilled, seconds * 16000)
        if (size == 0) return ShortArray(0)

        val result = ShortArray(size)
        val startPos = if (bufferFilled >= circularBuffer.size) bufferPosition else 0

        for (i in 0 until size) {
            result[i] = circularBuffer[(startPos + i) % circularBuffer.size]
        }
        return result
    }

    private fun recognizeAudio(samples: ShortArray): String {
        val recognizer = recognizer ?: return ""
        val stream = recognizer.createStream()
        val floatAudio = samples.map { it.toFloat() / 32768.0f }.toFloatArray()
        stream.acceptWaveform(floatAudio, 16000)
        recognizer.decode(stream)
        val result = recognizer.getResult(stream)
        return result.text.trim()
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        audioManager?.abandonAudioFocus(null)
        i( "Recording stopped")
    }

    private fun broadcastKeyword(keyword: String) {
        val intent = Intent(BROADCAST_KEYWORD).apply {
            putExtra(EXTRA_KEYWORD, keyword)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }
}
