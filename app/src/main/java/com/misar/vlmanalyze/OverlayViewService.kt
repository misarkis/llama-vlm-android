// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.localbroadcastmanager.content.LocalBroadcastManager

/**
 * Service that displays an overlay view over other apps.
 * Requires SYSTEM_ALERT_WINDOW permission.
 * Receives analysis results via BroadcastReceiver and displays them.
 */
class OverlayViewService : Service() {

    private var overlayView: View? = null
    private var windowManager: WindowManager? = null
    private var isShowing = false
    private var statusText: TextView? = null
    private var resultText: TextView? = null
    private var broadcastReceiver: BroadcastReceiver? = null
    private var lastResultText: String = ""
    private var lastStatusText: String = ""
    private var audioRecorder: AudioRecorder? = null
    private var isRecordingAudio = false
    private var currentAudioLevel = 0
    private val handler = Handler(Looper.getMainLooper())
    private var transcriber: SherpaTranscriber? = null
    private var isTranscribing = false

    // TTS state for tracking when speaking completes
    private var isSpeaking = false
    private var pendingResumeAfterTTS = false
    private var pendingTtsText: String? = null

    private var isStreaming = false  // Streaming state
    private var cameraSwitchBtn: ImageButton? = null
    private var prefs: android.content.SharedPreferences? = null

    // Continuous voice trigger state (circular buffer for 10 seconds of audio)
    private val circularBufferSize = 16000 * 10  // 10 seconds at 16kHz
    private val circularBuffer = ShortArray(circularBufferSize)
    private var circularBufferPos = 0
    private var circularBufferFilled = 0
    private var isVoiceTriggerEnabled = false
    private var isVoiceTriggerManuallyEnabled = false  // Track if user explicitly enabled mic
    private var lastTranscriptionTime = 0L
    private val TRANSCRIPTION_INTERVAL_MS = 500L  // Transcribe every 500ms
    private var voiceTriggerJob: Thread? = null
    private var isVoiceTriggerRunning = false
    private var isWaitingForVlmResponse = false  // Pause transcription during VLM analysis

    companion object {
        private const val TAG = "VLM-OverlayService"
        private const val MICROPHONE_NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "VLM_MICROPHONE_CHANNEL"

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        fun startService(context: Context) {
            i( "========== STARTING OVERLAY SERVICE ==========")
            val hasPermission = Settings.canDrawOverlays(context)
            i( "Overlay permission: $hasPermission")
            if (!hasPermission) {
                Log.e(TAG, "No overlay permission - cannot start")
                return
            }
            val intent = Intent(context, OverlayViewService::class.java)
            try {
                context.startService(intent)
                i( "startService called successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start service", e)
            }
        }

        fun stopService(context: Context) {
            i( "Stopping overlay service")
            context.stopService(Intent(context, OverlayViewService::class.java))
        }

        fun hideService(context: Context) {
            i( "Hiding overlay")
            val intent = Intent(context, OverlayViewService::class.java).apply {
                action = ACTION_HIDE_OVERLAY
            }
            try {
                context.startService(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to hide overlay", e)
            }
        }

        fun showService(context: Context) {
            i( "Showing overlay")
            val intent = Intent(context, OverlayViewService::class.java).apply {
                action = ACTION_SHOW_OVERLAY
            }
            try {
                context.startService(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to show overlay", e)
            }
        }

        private const val ACTION_HIDE_OVERLAY = "com.misar.vlmanalyze.HIDE_OVERLAY"
        private const val ACTION_SHOW_OVERLAY = "com.misar.vlmanalyze.SHOW_OVERLAY"

        // Microphone test broadcast
        private const val ACTION_MIC_TOGGLE = "com.misar.vlmanalyze.MIC_TOGGLE"

        // TTS completion broadcast
        private const val ACTION_TTS_COMPLETE = "com.misar.vlmanalyze.TTS_COMPLETE"

        // Camera switch broadcast
        private const val ACTION_CAMERA_SWITCH = "com.misar.vlmanalyze.CAMERA_SWITCH"
    }

    override fun onCreate() {
        super.onCreate()
        i( "Overlay service created")
        prefs = getSharedPreferences("VLM_Preferences", Context.MODE_PRIVATE)
        audioRecorder = AudioRecorder()
        transcriber = SherpaTranscriber(this)
        // Initialize transcriber in background to avoid blocking service creation
        Thread {
            val initialized = transcriber?.initialize()
            i( "Sherpa transcriber initialized: $initialized")
        }.start()
        createNotificationChannel()
        registerBroadcastReceiver()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        i( "========== Overlay service onStartCommand ===========")
        i( "Intent action: ${intent?.action}")
        when (intent?.action) {
            ACTION_HIDE_OVERLAY -> {
                i( "Hiding overlay on demand")
                hideOverlay()
            }
            ACTION_SHOW_OVERLAY -> {
                i( "Showing overlay on demand")
                showOverlay()
            }
            ACTION_MIC_TOGGLE -> {
                i( "Mic toggle from onStartCommand!")
                toggleAudioRecording()
            }
            else -> {
                showOverlay()
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        i( "Overlay service destroying")
        stopVoiceTriggerMonitoring()
        stopAudioRecording()  // Completely stop recorder and foreground service
        // Shut down TTS if active
        if (textToSpeech != null) {
            textToSpeech?.stop()
            textToSpeech?.shutdown()
            textToSpeech = null
        }
        hideOverlay()
        unregisterBroadcastReceiver()
        super.onDestroy()
    }

    private fun registerBroadcastReceiver() {
        broadcastReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                when (intent.action) {
                    ScreenCaptureService.ACTION_ANALYSIS_STREAM_COMPLETE -> {
                        // Streaming complete - speak and display full result
                        val content = intent.getStringExtra(ScreenCaptureService.EXTRA_STREAM_CONTENT) ?: ""
                        val promptTokens = intent.getIntExtra(ScreenCaptureService.EXTRA_RESULT_PROMPT_TOKENS, 0)
                        val completionTokens = intent.getIntExtra(ScreenCaptureService.EXTRA_RESULT_COMPLETION_TOKENS, 0)
                        val totalTimeMs = intent.getLongExtra(ScreenCaptureService.EXTRA_RESULT_TOTAL_TIME_MS, 0L)
                        val tokensPerSecond = intent.getDoubleExtra(ScreenCaptureService.EXTRA_RESULT_TOKENS_PER_SECOND, 0.0)

                        isStreaming = false

                        val cleanResult = cleanModelOutput(content)
                        lastResultText = cleanResult
                        val tokensTotal = promptTokens + completionTokens
                        val tps = if (tokensPerSecond > 0) tokensPerSecond.toString().take(4) + " t/s" else ""
                        val statsText = "\n\n---\nProcessing: ${totalTimeMs}ms | Tokens: ${tokensTotal} (prompt=${promptTokens}, completion=${completionTokens}) ${tps}"
                        val displayText = cleanResult + statsText

                        if (!isShowing) showOverlay()
                        updateResultText(displayText)
                        updateStatus("✅ Complete | $tokensTotal tokens | ${tps}")
                        i( "Stream complete: ${cleanResult.take(100)}... (native tps: $tokensPerSecond)")

                        // Speak the full result
                        speakAndResume(cleanResult)
                    }
                    ScreenCaptureService.ACTION_ANALYSIS_ERROR -> {
                        val error = intent.getStringExtra(ScreenCaptureService.EXTRA_ERROR_MESSAGE)
                        error?.let {
                            if (!isShowing) showOverlay()
                            updateResultText("Error: $it")
                            updateStatus("Error")
                            Log.e(TAG, "Analysis error: $it")

                            // Speak the error message (same as Camera mode)
                            speakAndResume("Error: $it")
                        }
                    }
                    ScreenCaptureService.ACTION_TRIGGER_CAPTURE -> {
                        // Capture started - status already set by button or voice trigger
                        i( "Capture triggered")
                    }
                    "com.misar.vlmanalyze.CAPTURE_STARTED" -> {
                        // Camera capture started - show overlay with analyzing status
                        i( "Camera capture started - showing analyzing status")
                        if (!isShowing) showOverlay()
                        updateStatus("🔍 Analyzing...")
                        updateResultText("Analyzing image...")
                    }
                    ScreenCaptureService.ACTION_CAPTURE_SAVED -> {
                        val imagePath = intent.getStringExtra(ScreenCaptureService.EXTRA_IMAGE_PATH)
                        imagePath?.let {
                            i( "Capture saved to: $it")
                        }
                        // Show overlay again after capture, status shows "Analyzing Screen..." until TTS starts
                        lastStatusText = "🔍 Analyzing Screen..."
                        showOverlay()
                    }
                    ACTION_MIC_TOGGLE -> {
                        i( "Mic toggle action received!")
                        toggleAudioRecording()
                    }
                }
            }
        }

        val filter = IntentFilter().apply {
            addAction(ScreenCaptureService.ACTION_ANALYSIS_STREAM_COMPLETE)
            addAction(ScreenCaptureService.ACTION_ANALYSIS_ERROR)
            addAction(ScreenCaptureService.ACTION_TRIGGER_CAPTURE)
            addAction(ScreenCaptureService.ACTION_CAPTURE_SAVED)
            addAction(ACTION_HIDE_OVERLAY)
            addAction(ACTION_SHOW_OVERLAY)
            addAction(ACTION_MIC_TOGGLE)
            addAction("com.misar.vlmanalyze.CAPTURE_STARTED")
        }

        // Use ContextCompat for Android 13+ compatibility
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            try {
                LocalBroadcastManager.getInstance(this).registerReceiver(broadcastReceiver!!, filter)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to register receiver", e)
            }
        } else {
            LocalBroadcastManager.getInstance(this).registerReceiver(broadcastReceiver!!, filter)
        }
        i( "BroadcastReceiver registered")
    }

    private fun unregisterBroadcastReceiver() {
        broadcastReceiver?.let {
            try {
                LocalBroadcastManager.getInstance(this).unregisterReceiver(it)
                i( "BroadcastReceiver unregistered")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to unregister receiver", e)
            }
        }
    }

    private fun toggleAudioRecording() {
        i( "Toggle audio recording: current state = $isRecordingAudio")
        if (isRecordingAudio) {
            stopAudioRecording()
        } else {
            startAudioRecording()
        }
    }

    private fun startAudioRecording() {
        val hasPermission = audioRecorder?.hasPermission(this)
        if (hasPermission != true) {
            updateStatus("Mic: Permission required")
            Log.e(TAG, "Microphone permission not granted")
            return
        }

        // Start as foreground service with microphone type
        startForeground(MICROPHONE_NOTIFICATION_ID, createMicrophoneNotification())

        // Reset circular buffer
        synchronized(circularBuffer) {
            circularBufferPos = 0
            circularBufferFilled = 0
        }
        lastTranscriptionTime = System.currentTimeMillis()

        isRecordingAudio = true
        isVoiceTriggerEnabled = true
        isVoiceTriggerManuallyEnabled = true  // User explicitly enabled mic
        startVoiceTriggerMonitoring()
        updateStatus("🎤 Mic: Voice trigger active")

        audioRecorder?.startRecording(this) { samples ->
            // Update circular buffer (thread-safe)
            updateCircularBuffer(samples)

            // Calculate audio level (RMS)
            var sum = 0L
            for (s in samples) {
                sum += kotlin.math.abs(s.toLong())
            }
            val avg = sum / samples.size
            currentAudioLevel = avg.toInt()

            // Update status with audio level on main thread
            handler.post {
                if (isRecordingAudio) {
                    statusText?.text = "🎤 Level: $currentAudioLevel"
                }
            }
        }
    }

    private fun updateCircularBuffer(samples: ShortArray) {
        // Stop writing to buffer if voice trigger is disabled
        if (!isVoiceTriggerEnabled) return

        synchronized(circularBuffer) {
            for (sample in samples) {
                circularBuffer[circularBufferPos] = sample
                circularBufferPos = (circularBufferPos + 1) % circularBufferSize
                if (circularBufferFilled < circularBufferSize) {
                    circularBufferFilled++
                }
            }
        }
    }

    private fun getRecentAudio(seconds: Int): ShortArray {
        val numSamples = minOf(seconds * 16000, circularBufferFilled)
        if (numSamples == 0) return ShortArray(0)

        val result = ShortArray(numSamples)
        synchronized(circularBuffer) {
            for (i in 0 until numSamples) {
                val pos = (circularBufferPos - numSamples + i + circularBufferSize) % circularBufferSize
                result[i] = circularBuffer[pos]
            }
        }
        return result
    }

    private fun startVoiceTriggerMonitoring() {
        if (isVoiceTriggerRunning) return
        isVoiceTriggerRunning = true
        isWaitingForVlmResponse = false

        voiceTriggerJob = Thread {
            i( "Voice trigger monitoring started")
            while (isVoiceTriggerRunning && isRecordingAudio) {
                try {
                    Thread.sleep(TRANSCRIPTION_INTERVAL_MS)

                    val now = System.currentTimeMillis()
                    if (now - lastTranscriptionTime >= TRANSCRIPTION_INTERVAL_MS) {
                        lastTranscriptionTime = now

                        // Get recent audio from circular buffer
                        val recentAudio = getRecentAudio(5)  // Last 5 seconds
                        if (recentAudio.isNotEmpty() && recentAudio.size >= 16000) {  // At least 1 second
                            // Check audio quality before transcription
                            var maxSample = 0L
                            for (s in recentAudio) {
                                val abs = kotlin.math.abs(s.toLong())
                                if (abs > maxSample) maxSample = abs
                            }

                            if (maxSample >= 100) {  // Minimum audio level
                                val text = transcriber?.transcribe(recentAudio.toList()) ?: ""

                                if (text.isNotBlank()) {
                                    i( "Voice recognized: '$text'")

                                    // Check for keywords BEFORE updating UI
                                    val keyword = VoiceKeywords.matchKeyword(text)
                                    if (keyword == "ANALYZE") {
                                        i( "ANALYZE keyword detected! Saving audio and triggering capture...")
                                        // Get recent audio for saving
                                        val audioToSave = getRecentAudio(5)
                                        // Stop polling and buffer writing immediately
                                        isVoiceTriggerRunning = false
                                        isVoiceTriggerEnabled = false
                                        // Clear buffer immediately
                                        synchronized(circularBuffer) {
                                            circularBufferPos = 0
                                            circularBufferFilled = 0
                                        }
                                        // Save audio in background thread
                                        if (Config.saveAudio && audioToSave.isNotEmpty()) {
                                            Thread {
                                                saveAudioWav(audioToSave.toList())
                                            }.start()
                                        }
                                        handler.post {
                                            isWaitingForVlmResponse = true
                                            pauseAudioRecording()  // Keep recorder running, stop buffer writing
                                            // Save status before hiding overlay
                                            lastStatusText = "🔍 Analyzing Screen..."
                                            hideOverlay()  // Hide to avoid appearing in capture
                                            triggerScreenCapture()
                                        }
                                        return@Thread
                                    } else if (keyword == "QUIT") {
                                        i( "QUIT keyword detected! Saving audio and stopping recording...")
                                        // Get recent audio for saving
                                        val audioToSave = getRecentAudio(5)
                                        isVoiceTriggerEnabled = false
                                        // Clear buffer immediately
                                        synchronized(circularBuffer) {
                                            circularBufferPos = 0
                                            circularBufferFilled = 0
                                        }
                                        // Save audio in background thread
                                        if (Config.saveAudio && audioToSave.isNotEmpty()) {
                                            Thread {
                                                saveAudioWav(audioToSave.toList())
                                            }.start()
                                        }
                                        handler.post {
                                            toggleAudioRecording()
                                        }
                                        return@Thread
                                    }

                                    // No keyword - update status with recognized speech
                                    handler.post {
                                        updateStatus("🗣️ '$text'")
                                    }
                                }
                            }
                        }
                    }
                } catch (e: InterruptedException) {
                    // Expected when stopping monitoring during normal operation (e.g., capture button click)
                    i( "Voice trigger monitoring interrupted")
                } catch (e: Exception) {
                    Log.e(TAG, "Voice trigger monitoring error: ${e.message}", e)
                }
            }
            i( "Voice trigger monitoring stopped")
        }.apply { start() }
    }

    private fun triggerScreenCapture() {
        try {
            // Broadcast via LocalBroadcastManager so both ScreenCaptureService and CameraCaptureActivity can receive
            val intent = Intent(ScreenCaptureService.ACTION_TRIGGER_CAPTURE)
            LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(intent)

            // Also start ScreenCaptureService for screen capture mode
            val serviceIntent = Intent(this@OverlayViewService, ScreenCaptureService::class.java).apply {
                action = ScreenCaptureService.ACTION_TRIGGER_CAPTURE
            }
            this@OverlayViewService.startService(serviceIntent)

            // Also send a standard broadcast for CameraCaptureActivity (in case it's active)
            val cameraIntent = Intent(CameraCaptureActivity.ACTION_CAMERA_CAPTURE)
            sendBroadcast(cameraIntent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to trigger capture: ${e.message}", e)
        }
    }

    private fun pauseAudioRecording() {
        // Just stop writing to buffer and monitoring - keep audio recorder running
        isRecordingAudio = false
        isVoiceTriggerEnabled = false
        stopVoiceTriggerMonitoring()
        currentAudioLevel = 0

        // Clear circular buffer
        synchronized(circularBuffer) {
            circularBufferPos = 0
            circularBufferFilled = 0
        }

        updateStatus("Mic: Paused")
        i( "Audio buffer writing paused (recorder still running)")
    }

    private fun stopAudioRecording() {
        // Completely stop the audio recorder and foreground service
        isVoiceTriggerManuallyEnabled = false  // User explicitly disabled mic
        isRecordingAudio = false  // Critical: mark as not recording
        stopForeground(STOP_FOREGROUND_REMOVE)
        audioRecorder?.stopRecording()
        i( "Audio recorder stopped completely")
    }

    private fun stopVoiceTriggerMonitoring() {
        isVoiceTriggerRunning = false
        isVoiceTriggerEnabled = false
        voiceTriggerJob?.interrupt()
        voiceTriggerJob?.join()
        voiceTriggerJob = null
        i( "Voice trigger monitoring stopped")
    }

    private fun resumeVoicePolling() {
        isWaitingForVlmResponse = false
        isRecordingAudio = true        // Re-enable audio recording flag
        isVoiceTriggerRunning = false  // Signal old thread to stop
        isVoiceTriggerEnabled = true   // Re-enable writing to buffer

        // Clear circular buffer to start fresh
        synchronized(circularBuffer) {
            circularBufferPos = 0
            circularBufferFilled = 0
        }
        lastTranscriptionTime = System.currentTimeMillis()

        // Only auto-resume monitoring if user manually enabled the mic
        if (isVoiceTriggerManuallyEnabled) {
            i( "Voice polling resumed - isRecordingAudio=true, buffer cleared, restarting monitoring")

            // Small delay to let old thread exit, then start new monitoring
            handler.postDelayed({
                startVoiceTriggerMonitoring()
            }, 100)

            // Update status to show mic is active (clears any previous status like "Speaking...")
            handler.post {
                updateStatus("🎤 Mic: Voice trigger active")
            }
        } else {
            i( "Voice polling resumed (manual mode) - isRecordingAudio=true, buffer cleared, monitoring NOT restarted")
            // Update status to show mic is available but not active
            handler.post {
                updateStatus("🎤 Mic: Tap to enable voice")
            }
        }
    }

    private fun transcribeAudio(samples: List<Short>) {
        if (isTranscribing) {
            Log.w(TAG, "Already transcribing, skipping")
            return
        }
        isTranscribing = true

        // Transcribe in background thread to avoid blocking UI
        Thread {
            try {
                val text = transcriber?.transcribe(samples) ?: ""

                handler.post {
                    if (text.isNotBlank()) {
                        updateStatus("🎤 '$text'")
                        updateResultText("Voice: $text")
                        i( "Transcription result: $text")

                        // Check for keywords after transcription
                        val keyword = VoiceKeywords.matchKeyword(text)
                        if (keyword != null) {
                            i( "Keyword detected: $keyword from '$text'")
                            if (keyword == "QUIT") {
                                toggleAudioRecording()
                            }
                        }
                    } else {
                        updateStatus("Mic: No speech recognized")
                        Log.w(TAG, "No speech recognized from audio")
                    }
                    isTranscribing = false
                }
            } catch (e: Exception) {
                handler.post {
                    updateStatus("Mic: Transcription error")
                    Log.e(TAG, "Transcription error: ${e.message}", e)
                    isTranscribing = false
                }
            }
        }.start()
    }

    private var textToSpeech: android.speech.tts.TextToSpeech? = null
    private var ttsTimerRunnable: Runnable? = null

    private fun cancelTtsTimer() {
        ttsTimerRunnable?.let { handler.removeCallbacks(it) }
        ttsTimerRunnable = null
    }

    // Clean up garbage tokens from model output - stops at first garbage token
    private fun cleanModelOutput(text: String): String {
        // Stop at garbage tokens like <|start_header_id|>system<|end_header_id|>
        val garbagePatterns = listOf(
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|start_header|>",
            "<|end_header|>",
            "<|box_end|>",
            "<|box_start|>"
        )
        var cleaned = text
        for (pattern in garbagePatterns) {
            val idx = cleaned.indexOf(pattern)
            if (idx != -1) {
                cleaned = cleaned.substring(0, idx)
                break
            }
        }
        return cleaned.trim()
    }

    private fun speakAndResume(text: String) {
        pendingTtsText = text
        // Cancel any existing timer
        cancelTtsTimer()

        // Use Android TTS to speak the result
        textToSpeech = android.speech.tts.TextToSpeech(this) { status ->
            if (status == android.speech.tts.TextToSpeech.SUCCESS && pendingTtsText != null) {
                isSpeaking = true
                textToSpeech?.setSpeechRate(1.2f)

                // Use UtteranceProgressListener - requires params HashMap with utterance ID
                val params = HashMap<String, String>()
                val utteranceId = "vlm_speak_${System.currentTimeMillis()}"
                params["android.speech.extra.UTTERANCE_ID"] = utteranceId

                textToSpeech?.speak(pendingTtsText, android.speech.tts.TextToSpeech.QUEUE_ADD, params)

                // Set UtteranceProgressListener (modern API, replaces deprecated setOnUtteranceCompletedListener)
                textToSpeech?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(id: String?) {
                        d("TTS utterance started: $id")
                    }
                    override fun onDone(id: String?) {
                        i( "TTS utterance complete: $id")
                        cancelTtsTimer()
                        isSpeaking = false
                        textToSpeech?.shutdown()
                        textToSpeech = null
                        pendingTtsText = null
                        handler.post {
                            // Broadcast TTS complete to CameraCaptureActivity so it can resume camera stream
                            val intent = Intent("com.misar.vlmanalyze.TTS_COMPLETE")
                            LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(intent)
                            resumeVoicePolling()  // Auto-resume based on isVoiceTriggerManuallyEnabled flag
                        }
                    }
                    override fun onError(id: String?) {
                        Log.e(TAG, "TTS error: $id")
                        cancelTtsTimer()
                        isSpeaking = false
                        textToSpeech?.shutdown()
                        textToSpeech = null
                        pendingTtsText = null
                        handler.post {
                            resumeVoicePolling()  // Auto-resume on error
                        }
                    }
                })

                // Fallback timer: estimate speech duration from text length
                // At 1.2x speech rate, ~15-20 characters per second
                val estimatedDurationMs = (text.length * 1000L / 18).coerceAtLeast(3000) + 2000
                i( "TTS fallback timer set for ${estimatedDurationMs}ms (text length: ${text.length})")
                ttsTimerRunnable = Runnable {
                    if (isSpeaking) {
                        Log.w(TAG, "TTS fallback timer fired - broadcasting TTS complete and resuming")
                        isSpeaking = false
                        textToSpeech?.shutdown()
                        textToSpeech = null
                        pendingTtsText = null
                        // Broadcast TTS complete to CameraCaptureActivity
                        val intent = Intent("com.misar.vlmanalyze.TTS_COMPLETE")
                        LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(intent)
                        resumeVoicePolling()  // Auto-resume based on isVoiceTriggerManuallyEnabled flag
                    }
                }
                handler.postDelayed(ttsTimerRunnable!!, estimatedDurationMs)
            } else {
                Log.e(TAG, "TTS initialization failed")
                handler.post {
                    resumeVoicePolling()  // Auto-resume on init failure
                }
            }
        }
    }

    private fun saveAudioWav(samples: List<Short>) {
        try {
            val timestamp = System.currentTimeMillis()
            val fileName = "VLM_AUDIO_${timestamp}.wav"

            // Save to Downloads/VLM_Capture using MediaStore (Android 10+)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                val resolver = contentResolver
                val values = android.content.ContentValues().apply {
                    put(android.provider.MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(android.provider.MediaStore.Downloads.MIME_TYPE, "audio/wav")
                    put(android.provider.MediaStore.Downloads.RELATIVE_PATH, "${android.os.Environment.DIRECTORY_DOWNLOADS}/VLM_Capture")
                }

                val uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    resolver.openOutputStream(uri)?.use { outputStream ->
                        writeWavFile(outputStream, samples)
                        val path = "Downloads/VLM_Capture/$fileName"
                        i( "Saved audio to public Downloads: $path")
                        // No status update - saving happens silently in background
                    }
                }
            } else {
                // Fallback for Android 9 and below - use file system
                val dir = java.io.File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS), "VLM_Capture")
                if (!dir.exists()) dir.mkdirs()
                val file = java.io.File(dir, fileName)
                file.outputStream().use { os ->
                    writeWavFile(os, samples)
                }
                i( "Saved audio to ${file.absolutePath}")
                // No status update - saving happens silently in background
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save audio: ${e.message}", e)
        }
    }

    private fun writeWavFile(outputStream: java.io.OutputStream, samples: List<Short>) {
        val sampleRate = 16000
        val bitsPerSample = 16
        val channels = 1
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = samples.size * 2
        val totalDataLen = dataSize + 36

        // Write WAV header byte by byte (little-endian)
        val header = ByteArray(44)

        // RIFF chunk
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()

        // fmt subchunk
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16 // Subchunk1Size (16 for PCM)
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // AudioFormat (1 = PCM)
        header[21] = 0
        header[22] = channels.toByte() // NumChannels
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = blockAlign.toByte() // BlockAlign
        header[33] = 0
        header[34] = bitsPerSample.toByte() // BitsPerSample
        header[35] = 0

        // data subchunk
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (dataSize and 0xff).toByte()
        header[41] = ((dataSize shr 8) and 0xff).toByte()
        header[42] = ((dataSize shr 16) and 0xff).toByte()
        header[43] = ((dataSize shr 24) and 0xff).toByte()

        // Write header
        outputStream.write(header)

        // Write audio data (PCM16 samples in little-endian)
        samples.forEach { sample ->
            val s = sample.toInt()
            outputStream.write(s and 0xff)
            outputStream.write((s shr 8) and 0xff)
        }

        outputStream.flush()
    }

    private fun updateStatus(status: String) {
        lastStatusText = status  // Always save last status
        if (isShowing) {
            statusText?.text = status
        }
    }

    private fun updateResultText(text: String) {
        if (isShowing) {
            resultText?.text = text
        }
    }

    private fun showOverlay() {
        if (isShowing) {
            Log.w(TAG, "Overlay already showing")
            return
        }

        try {
            i( "Creating overlay view")
            overlayView = createOverlayView()
            i( "Overlay view created, getting window manager")

            windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
            val params = getWindowParams()
            i( "Window params: type=${params.type}, width=${params.width}, height=${params.height}, x=${params.x}, y=${params.y}")

            i( "Adding view to window manager")
            windowManager?.addView(overlayView, params)
            isShowing = true
            i( "SUCCESS: Overlay shown successfully, isShowing=$isShowing")

            // Restore last result if any
            if (lastResultText.isNotEmpty()) {
                resultText?.text = lastResultText
            }

            // Restore last status if any
            if (lastStatusText.isNotEmpty()) {
                statusText?.text = lastStatusText
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "SecurityException - overlay permission may not be granted", e)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to show overlay", e)
        }
    }

    private fun hideOverlay() {
        if (!isShowing) return
        try {
            overlayView?.let {
                windowManager?.removeView(it)
            }
            overlayView = null
            isShowing = false
            i( "Overlay hidden")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to hide overlay", e)
        }
    }

    private fun createOverlayView(): View {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#4d1a1a1a"))
            setPadding(16, 16, 16, 16)
            gravity = Gravity.CENTER
        }

        statusText = TextView(this).apply {
            id = View.generateViewId()
            text = "Overlay active"
            setTextColor(Color.parseColor("#888888"))
            textSize = 12f
            setBackgroundColor(Color.parseColor("#333333"))
            setPadding(8, 8, 8, 8)
            gravity = Gravity.CENTER
        }

        val buttonContainer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        val micBtn = ImageButton(this).apply {
            setImageResource(R.drawable.ic_microphone)
            setBackgroundResource(android.R.color.transparent)
            imageTintList = android.content.res.ColorStateList.valueOf(Color.WHITE)
            scaleType = android.widget.ImageView.ScaleType.CENTER_CROP
            setOnClickListener {
                i( "Mic button clicked!")
                try {
                    val intent = Intent(this@OverlayViewService, OverlayViewService::class.java).apply {
                        action = ACTION_MIC_TOGGLE
                    }
                    this@OverlayViewService.startService(intent)
                    i( "Mic toggle intent sent")
                } catch (e: Exception) {
                    Log.e(TAG, "Mic button error: ${e.message}", e)
                    updateStatus("Mic error: ${e.message}")
                }
            }
        }

        val captureBtn = ImageButton(this).apply {
            setImageResource(R.drawable.ic_capture)
            setBackgroundResource(android.R.color.transparent)
            imageTintList = android.content.res.ColorStateList.valueOf(Color.WHITE)
            scaleType = android.widget.ImageView.ScaleType.CENTER_CROP
            setOnClickListener {
                i( "Capture button clicked")
                try {
                    // Pause voice polling and save status before capture
                    pauseAudioRecording()
                    lastStatusText = "🔍 Analyzing..."

                    // Broadcast CAPTURE_STARTED first to show "Analyzing..." status immediately
                    val captureStartedIntent = Intent("com.misar.vlmanalyze.CAPTURE_STARTED")
                    LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(captureStartedIntent)

                    // Broadcast to ScreenCaptureService (screen mode)
                    val triggerIntent = Intent(ScreenCaptureService.ACTION_TRIGGER_CAPTURE)
                    LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(triggerIntent)

                    // Start ScreenCaptureService for screen mode
                    val serviceIntent = Intent(this@OverlayViewService, ScreenCaptureService::class.java).apply {
                        action = ScreenCaptureService.ACTION_TRIGGER_CAPTURE
                    }
                    this@OverlayViewService.startService(serviceIntent)

                    // Broadcast to CameraCaptureActivity for camera mode
                    val cameraIntent = Intent(CameraCaptureActivity.ACTION_CAMERA_CAPTURE)
                    i("Sending LocalBroadcast to CameraCaptureActivity: ${CameraCaptureActivity.ACTION_CAMERA_CAPTURE}")
                    LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(cameraIntent)
                    i("LocalBroadcast sent to CameraCaptureActivity")
                } catch (e: Exception) {
                    updateStatus("Error: ${e.message}")
                }
            }
        }

        buttonContainer.addView(micBtn, LinearLayout.LayoutParams(154, 154))

        // Capture button with spacing after mic
        val captureParams = LinearLayout.LayoutParams(154, 154)
        captureParams.leftMargin = 32
        buttonContainer.addView(captureBtn, captureParams)

        // Camera switch button - only show when capture source is Camera
        val captureSource = prefs?.getString("capture_source", "SCREEN") ?: "SCREEN"
        if (captureSource == "CAMERA") {
            cameraSwitchBtn = ImageButton(this).apply {
                setImageResource(R.drawable.ic_switch_camera)
                setBackgroundResource(android.R.color.transparent)
                imageTintList = android.content.res.ColorStateList.valueOf(Color.WHITE)
                scaleType = android.widget.ImageView.ScaleType.CENTER_CROP
                setOnClickListener {
                    i("Camera switch button clicked in overlay")
                    try {
                        // Send LocalBroadcast to CameraCaptureActivity
                        val intent = Intent(ACTION_CAMERA_SWITCH)
                        LocalBroadcastManager.getInstance(this@OverlayViewService).sendBroadcast(intent)
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera switch button error: ${e.message}", e)
                    }
                }
            }
            val switchParams = LinearLayout.LayoutParams(154, 154)
            switchParams.leftMargin = 32
            buttonContainer.addView(cameraSwitchBtn, switchParams)
        }

        container.addView(statusText!!)
        container.addView(buttonContainer)

        setupDrag(container)
        return container
    }

    private fun dipToPixels(dip: Int): Int {
        return (dip * resources.displayMetrics.density + 0.5f).toInt()
    }

    private fun getWindowParams(): WindowManager.LayoutParams {
        return WindowManager.LayoutParams().apply {
            type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                WindowManager.LayoutParams.TYPE_PHONE
            }
            format = PixelFormat.RGBA_8888
            flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
            width = WindowManager.LayoutParams.WRAP_CONTENT
            height = WindowManager.LayoutParams.WRAP_CONTENT
            gravity = Gravity.BOTTOM or Gravity.START
            x = 20   // 20px from left edge
            y = 20   // 20px from bottom
        }
    }

    private fun setupDrag(view: View) {
        var initialX = 0f
        var initialY = 0f
        var origX = 0
        var origY = 0

        view.setOnTouchListener(object : View.OnTouchListener {
            override fun onTouch(v: View?, event: MotionEvent): Boolean {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = event.rawX
                        initialY = event.rawY
                        val params = overlayView?.layoutParams as? WindowManager.LayoutParams
                        origX = params?.x ?: 0
                        origY = params?.y ?: 0
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val params = overlayView?.layoutParams as? WindowManager.LayoutParams ?: return true
                        val dx = (event.rawX - initialX).toInt()
                        val dy = (initialY - event.rawY).toInt()
                        params.x = origX + dx
                        params.y = origY + dy
                        windowManager?.updateViewLayout(v, params)
                    }
                }
                return true
            }
        })
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "VLM Microphone",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Microphone active for voice commands"
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createMicrophoneNotification(): Notification {
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VLM Microphone Active")
            .setContentText("Listening for voice commands")
            .setSmallIcon(android.R.drawable.ic_menu_call)
            .setOngoing(true)

        return builder.build()
    }
}
