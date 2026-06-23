// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.WindowManager
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer

/**
 * Foreground service for screen capture.
 * Uses either remote API or local inference based on backend mode.
 */
class ScreenCaptureService : Service() {

    private var mediaProjection: MediaProjection? = null
    private val handler = Handler(Looper.getMainLooper())
    private var imageReader: ImageReader? = null
    private var isRunning = false
    private var displayCallback: MediaProjection.Callback? = null
    private val serviceScope = CoroutineScope(Dispatchers.Default + Job())

    private val vlmAnalyzer = VlmAnalyzer()
    private var lastCaptureTime = 0L
    private var captureCount = 0

    // Backend mode (now stored statically in companion object to persist across Intent calls)
    // Note: For LOCAL mode, we use VlmApplication.localInferenceBackend (singleton)
    private val backendMode: String get() = sBackendMode

    companion object {
        private const val TAG = "VLM-ScrCaptureSvc"
        const val LOG_TAG_SERVICE = "VLM-Service"
        const val LOG_TAG_CAPTURE = "VLM-Capture"
        const val LOG_TAG_INFERENCE = "VLM-Inference"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "vlm_capture_channel"

        // Debug log helper - only logs when debug mode is enabled
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        // Info log helpers for different sub-components
        private fun iService(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(LOG_TAG_SERVICE, msg)
        }

        private fun iCapture(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(LOG_TAG_CAPTURE, msg)
        }

        private fun iInference(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(LOG_TAG_INFERENCE, msg)
        }

        // Backend mode constant
        const val EXTRA_BACKEND_MODE = "backend_mode"

        // Track service state for triggerCapture (set in instance, read statically)
        @Volatile
        private var sIsRunning = false

        // Persist backend mode across Intent calls (set once at service start)
        @Volatile
        private var sBackendMode: String = "REMOTE"

        fun isServiceRunning(): Boolean = sIsRunning

        fun startService(context: Context, resultCode: Int, data: Intent) {
            i( "startService called with resultCode=$resultCode, data!=null=${data != null}")
            // Just pass the Intent directly - it already contains the backend mode
            context.startForegroundService(data)
            i( "startForegroundService called")
        }

        fun triggerCapture(context: Context) {
            i( "triggerCapture called")
            i( "  sIsRunning=$sIsRunning, pendingData=${VlmApplication.pendingData != null}")

            if (!sIsRunning || VlmApplication.pendingData == null) {
                Log.w(TAG, "Service not ready - sIsRunning=$sIsRunning, hasProjectionData=${VlmApplication.pendingData != null}")
                return
            }

            val intent = Intent(context, ScreenCaptureService::class.java).apply {
                action = ACTION_TRIGGER_CAPTURE
            }
            try {
                context.startService(intent)
                i( "triggerCapture sent successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send triggerCapture", e)
            }
        }

        fun stopService(context: Context) {
            i( "stopService called")
            context.stopService(Intent(context, ScreenCaptureService::class.java))
        }

        const val ACTION_TRIGGER_CAPTURE = "com.misar.vlmanalyze.TRIGGER_CAPTURE"
        private const val ACTION_STOP_SERVICE = "com.misar.vlmanalyze.STOP_SERVICE"

        // Broadcast actions for results
        const val ACTION_ANALYSIS_RESULT = "com.misar.vlmanalyze.ANALYSIS_RESULT"
        const val ACTION_ANALYSIS_STREAM_TOKEN = "com.misar.vlmanalyze.ANALYSIS_STREAM_TOKEN"
        const val ACTION_ANALYSIS_STREAM_COMPLETE = "com.misar.vlmanalyze.ANALYSIS_STREAM_COMPLETE"
        const val ACTION_ANALYSIS_ERROR = "com.misar.vlmanalyze.ANALYSIS_ERROR"
        const val ACTION_CAPTURE_SAVED = "com.misar.vlmanalyze.CAPTURE_SAVED"
        const val EXTRA_RESULT_CONTENT = "result_content"
        const val EXTRA_RESULT_PROMPT_TOKENS = "prompt_tokens"
        const val EXTRA_RESULT_COMPLETION_TOKENS = "completion_tokens"
        const val EXTRA_RESULT_TOTAL_TIME_MS = "total_time_ms"
        const val EXTRA_RESULT_TOKENS_PER_SECOND = "tokens_per_second"
        const val EXTRA_ERROR_MESSAGE = "error_message"
        const val EXTRA_IMAGE_PATH = "image_path"
        const val EXTRA_STREAM_TOKEN = "stream_token"
        const val EXTRA_STREAM_CONTENT = "stream_content"
        const val EXTRA_STREAMING_COMPLETE = "streaming_complete"
    }

    override fun onCreate() {
        super.onCreate()
        i( "Service created")

        // Initialize VLM analyzer
        vlmAnalyzer.initialize(filesDir.absolutePath)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        iService( "===== Service onStartCommand: ${intent?.action} =====")

        // Get backend mode from Intent (only set on initial start, not on trigger captures)
        if (intent != null) {
            val incomingMode = intent.getStringExtra(EXTRA_BACKEND_MODE) ?: "REMOTE"
            // Only update sBackendMode if this is the initial start (not a trigger action)
            if (intent.action != ACTION_TRIGGER_CAPTURE && sBackendMode == "REMOTE") {
                sBackendMode = incomingMode
                iService( "===== Backend mode set to: $sBackendMode =====")
            } else if (intent.action == ACTION_TRIGGER_CAPTURE) {
                iService( "===== Trigger capture - using existing backend mode: $sBackendMode =====")
            }
        }

        when (intent?.action) {
            ACTION_TRIGGER_CAPTURE -> {
                if (isRunning && VlmApplication.pendingData != null) {
                    captureSingleFrame()
                } else {
                    Log.w(TAG, "Service not ready - isRunning=$isRunning")
                }
            }
            ACTION_STOP_SERVICE -> {
                i( "Stop requested from notification")
                stopSelf()
            }
            else -> {
                i( "Initial start - getting projection data from Intent")
                i( "  intent!=null: ${intent != null}")
                i( "  resultCode extra: ${intent?.getIntExtra("resultCode", -999)}")
                i( "  media_projection_data extra: ${intent?.getParcelableExtra<Intent>("media_projection_data") != null}")

                // Get media projection data from Intent extras
                var resultCode = intent?.getIntExtra("resultCode", -1) ?: -1
                @Suppress("DEPRECATION")
                var data: Intent? = intent?.getParcelableExtra<Intent>("media_projection_data")

                i( "  After extraction - resultCode=$resultCode, data!=null=${data != null}")

                if (data == null || resultCode == -1) {
                    Log.e(TAG, "No projection data in Intent: resultCode=$resultCode, data=${data}")
                    Log.e(TAG, "  Fallback: checking VlmApplication.pendingData=${VlmApplication.pendingData != null}")

                    // Fallback: use persisted data from VlmApplication if available
                    if (VlmApplication.pendingData != null && VlmApplication.pendingResult != null) {
                        i( "  Using fallback from VlmApplication: resultCode=${VlmApplication.pendingResult}")
                        resultCode = VlmApplication.pendingResult!!
                        data = VlmApplication.pendingData!!
                    } else {
                        createNotificationChannel()
                        val notification = Notification.Builder(this, CHANNEL_ID)
                            .setContentTitle("VLM Analyze")
                            .setContentText("Error: No projection data")
                            .setSmallIcon(android.R.drawable.ic_menu_info_details)
                            .build()
                        startForeground(NOTIFICATION_ID, notification)
                        stopSelf()
                        return START_NOT_STICKY
                    }
                }

                createNotificationChannel()

                val notification = Notification.Builder(this, CHANNEL_ID)
                    .setContentTitle("VLM Analyze")
                    .setContentText("Starting...")
                    .setSmallIcon(android.R.drawable.ic_menu_info_details)
                    .setOngoing(true)
                    .build()

                startForeground(NOTIFICATION_ID, notification)
                i( "startForeground completed")

                setupProjection(resultCode, data)
            }
        }

        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        i( "===== Service onDestroy =====")
        isRunning = false
        sIsRunning = false
        handler.removeCallbacksAndMessages(null)
        imageReader?.close()

        val callback = displayCallback
        if (callback != null) {
            mediaProjection?.unregisterCallback(callback)
            displayCallback = null
        }
        mediaProjection?.stop()
        mediaProjection = null

        VlmApplication.pendingResult = null
        VlmApplication.pendingData = null
        sBackendMode = "REMOTE"  // Reset backend mode
        i( "Cleared VlmApplication singleton data")

        vlmAnalyzer.destroy()
        serviceScope.cancel()

        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            i( "Creating notification channel")
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Screen Capture",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
            i( "Notification channel created")
        }
    }

    private fun setupProjection(resultCode: Int, data: Intent) {
        val captureIntent = Intent(this, ScreenCaptureService::class.java).apply {
            action = ACTION_TRIGGER_CAPTURE
        }
        val capturePending = PendingIntent.getService(
            this, 0, captureIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, ScreenCaptureService::class.java).apply {
            action = ACTION_STOP_SERVICE
        }
        val stopPending = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val mainIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val mainPending = PendingIntent.getActivity(
            this, 2, mainIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VLM Analyze")
            .setContentText("Tap Capture to analyze screen")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .setContentIntent(mainPending)
            .addAction(android.R.drawable.ic_menu_camera, "Capture", capturePending)
            .addAction(android.R.drawable.ic_menu_delete, "Stop", stopPending)
            .build()

        startForeground(NOTIFICATION_ID, notification)

        try {
            val mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, data)

            setupCaptureReader()
            isRunning = true
            sIsRunning = true
            i( "Service ready for captures")

        } catch (e: Exception) {
            Log.e(TAG, "Error setting up projection", e)
            stopSelf()
        }
    }

    private fun setupCaptureReader() {
        val windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val display = windowManager.defaultDisplay
        val displayMetrics = android.util.DisplayMetrics()
        display.getRealMetrics(displayMetrics)

        val width = displayMetrics.widthPixels
        val height = displayMetrics.heightPixels
        val densityDpi = displayMetrics.densityDpi

        i( "Setup reader: ${width}x${height}")

        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)

        val callback = object : MediaProjection.Callback() {
            override fun onStop() {
                i( "MediaProjection stopped")
                isRunning = false
            }
        }
        displayCallback = callback

        mediaProjection?.registerCallback(callback, handler)

        mediaProjection?.createVirtualDisplay(
            "VLM Capture",
            width, height, densityDpi,
            0,
            imageReader?.surface,
            null,
            handler
        )
    }

    private fun captureSingleFrame() {
        iCapture( "========== captureSingleFrame START ==========")

        handler.postDelayed({
            try {
                iCapture( "Hiding overlay before capture")
                // Hide overlay before capture to avoid it appearing in screenshot
                OverlayViewService.hideService(this)

                // Wait 500ms for overlay to disappear before capturing
                handler.postDelayed({
                    iCapture( "Calling doCaptureAndAnalyze")
                    doCaptureAndAnalyze()
                    // Restore overlay immediately after capture starts (not after analysis)
                    // This allows the overlay to show status during TTS
                    handler.post {
                        iCapture( "Restoring overlay after capture")
                        OverlayViewService.showService(this@ScreenCaptureService)
                    }
                }, 500)
                return@postDelayed
            } catch (e: Exception) {
                Log.e(LOG_TAG_CAPTURE, "Capture error", e)
            }
        }, 100)
    }

    private fun doCaptureAndAnalyze() {
        iCapture( "doCaptureAndAnalyze START")
        val image = imageReader?.acquireLatestImage()
        if (image != null) {
            val origWidth = image.width
            val origHeight = image.height
            iCapture( "Image acquired: ${origWidth}x${origHeight}, scale=${Config.imageScaleFactorScreen}")

            // Apply image scale factor
            val scaledWidth = (origWidth * Config.imageScaleFactorScreen).toInt().coerceAtLeast(1)
            val scaledHeight = (origHeight * Config.imageScaleFactorScreen).toInt().coerceAtLeast(1)
            iCapture( "Scaling to: ${scaledWidth}x${scaledHeight}")

            val plane = image.planes[0]
            val buffer = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride

            // Check backend mode to decide encoding method
            val isLocalMode = backendMode == "LOCAL"

            if (isLocalMode) {
                // LOCAL mode: use RGB bytes directly (no JPEG compression)
                val rgbBytes = rgbaToRgbBytesScaled(buffer, origWidth, origHeight, rowStride, pixelStride, scaledWidth, scaledHeight)
                image.close()

                if (rgbBytes.isEmpty()) {
                    Log.e(TAG, "Failed to encode image")
                    sendError("Image encoding failed")
                    return
                }

                iCapture( "RGB bytes length: ${rgbBytes.size}")

                // Analyze with VLM
                serviceScope.launch(Dispatchers.IO) {
                    analyzeScreenWithRgbBytes(rgbBytes, scaledWidth, scaledHeight)
                }
            } else {
                // REMOTE mode: use JPEG base64
                val base64Image = rgbaToJpegBase64Scaled(buffer, origWidth, origHeight, rowStride, pixelStride, scaledWidth, scaledHeight)
                image.close()

                if (base64Image.isEmpty()) {
                    Log.e(TAG, "Failed to encode image")
                    sendError("Image encoding failed")
                    return
                }

                iCapture( "Base64 length: ${base64Image.length}")

                // Analyze with VLM
                serviceScope.launch(Dispatchers.IO) {
                    analyzeScreen(base64Image)
                }
            }
        } else {
            Log.w(LOG_TAG_CAPTURE, "No image available from imageReader")
        }
    }

    private fun rgbaToJpegBase64(buffer: ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int): String {
        try {
            // Create array to hold properly packed pixel data
            val pixels = IntArray(width * height)

            // Read pixel data row by row, accounting for rowStride and pixelStride
            // Buffer is RGBA, but Android Bitmap expects ARGB (swap R and B)
            buffer.rewind()
            for (y in 0 until height) {
                val rowStart = y * width
                for (x in 0 until width) {
                    // Calculate byte position: row * rowStride + column * pixelStride
                    val bytePos = y * rowStride + x * pixelStride
                    buffer.position(bytePos)

                    // Read RGBA bytes
                    val r = buffer.get().toInt() and 0xFF
                    val g = buffer.get().toInt() and 0xFF
                    val b = buffer.get().toInt() and 0xFF
                    val a = buffer.get().toInt() and 0xFF

                    // Pack as ARGB (Android format): A in highest byte, then R, G, B
                    pixels[rowStart + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            // Create bitmap from packed pixel data
            val fullBitmap = android.graphics.Bitmap.createBitmap(pixels, width, height, android.graphics.Bitmap.Config.ARGB_8888)

            // Apply scale factor if less than 1.0
            val bitmap = if (Config.imageScaleFactorScreen < 1.0f) {
                val scaledWidth = (width * Config.imageScaleFactorScreen).toInt().coerceAtLeast(1)
                val scaledHeight = (height * Config.imageScaleFactorScreen).toInt().coerceAtLeast(1)
                android.graphics.Bitmap.createScaledBitmap(fullBitmap, scaledWidth, scaledHeight, true)
            } else {
                fullBitmap
            }

            val outputStream = ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, Config.imageQuality, outputStream)
            fullBitmap.recycle()
            if (bitmap != fullBitmap) {
                bitmap.recycle()
            }

            return android.util.Base64.encodeToString(outputStream.toByteArray(), android.util.Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert RGBA to JPEG", e)
            return ""
        }
    }

    /**
     * Convert RGBA buffer to RGB byte array with scaling applied.
     * Used for LOCAL backend mode to avoid JPEG compression/decompression overhead.
     *
     * @return RGB byte array (width * height * 3 bytes)
     */
    private fun rgbaToRgbBytesScaled(buffer: ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int, scaledWidth: Int, scaledHeight: Int): ByteArray {
        try {
            // Create full size bitmap
            val fullBitmap = android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888)

            // Read RGBA bytes into pixel array
            val pixels = IntArray(width * height)
            buffer.rewind()
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val bytePos = y * rowStride + x * pixelStride
                    buffer.position(bytePos)
                    val r = buffer.get().toInt() and 0xFF
                    val g = buffer.get().toInt() and 0xFF
                    val b = buffer.get().toInt() and 0xFF
                    val a = buffer.get().toInt() and 0xFF
                    pixels[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            fullBitmap.setPixels(pixels, 0, width, 0, 0, width, height)

            // Scale down if needed
            val bitmap = if (scaledWidth < width || scaledHeight < height) {
                android.graphics.Bitmap.createScaledBitmap(fullBitmap, scaledWidth, scaledHeight, true)
            } else {
                fullBitmap
            }

            // Convert to RGB byte array (3 bytes per pixel: R, G, B)
            val rgbBytes = ByteArray(scaledWidth * scaledHeight * 3)
            val pixelArray = IntArray(scaledWidth * scaledHeight)
            bitmap.getPixels(pixelArray, 0, scaledWidth, 0, 0, scaledWidth, scaledHeight)

            for (i in pixelArray.indices) {
                val pixel = pixelArray[i]
                rgbBytes[i * 3] = ((pixel shr 16) and 0xFF).toByte()
                rgbBytes[i * 3 + 1] = ((pixel shr 8) and 0xFF).toByte()
                rgbBytes[i * 3 + 2] = (pixel and 0xFF).toByte()
            }

            fullBitmap.recycle()
            if (bitmap != fullBitmap) {
                bitmap.recycle()
            }

            d("RGB bytes: ${rgbBytes.size} bytes (${scaledWidth}x${scaledHeight})")
            return rgbBytes
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert RGBA to RGB bytes", e)
            return ByteArray(0)
        }
    }

    /**
     * Convert RGBA buffer to JPEG base64 with scaling applied.
     * Uses Android Bitmap for scaling and compression.
     */
    private fun rgbaToJpegBase64Scaled(buffer: ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int, scaledWidth: Int, scaledHeight: Int): String {
        try {
            // Create full size bitmap first
            val fullBitmap = android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888)

            // Read RGBA bytes into pixel array
            val pixels = IntArray(width * height)
            buffer.rewind()
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val bytePos = y * rowStride + x * pixelStride
                    buffer.position(bytePos)
                    val r = buffer.get().toInt() and 0xFF
                    val g = buffer.get().toInt() and 0xFF
                    val b = buffer.get().toInt() and 0xFF
                    val a = buffer.get().toInt() and 0xFF
                    pixels[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            fullBitmap.setPixels(pixels, 0, width, 0, 0, width, height)

            // Scale down if needed
            val bitmap = if (scaledWidth < width || scaledHeight < height) {
                android.graphics.Bitmap.createScaledBitmap(fullBitmap, scaledWidth, scaledHeight, true)
            } else {
                fullBitmap
            }

            val outputStream = ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, Config.imageQuality, outputStream)
            fullBitmap.recycle()
            if (bitmap != fullBitmap) {
                bitmap.recycle()
            }

            return android.util.Base64.encodeToString(outputStream.toByteArray(), android.util.Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert RGBA to JPEG with scaling", e)
            return ""
        }
    }

    private suspend fun analyzeScreenWithRgbBytes(rgbBytes: ByteArray, width: Int, height: Int) {
        iInference( "===== analyzeScreenWithRgbBytes START =====")
        iInference( "Backend mode: $backendMode")
        iInference( "Image: ${width}x${height}, RGB bytes: ${rgbBytes.size}")

        // Check cooldown
        val now = System.currentTimeMillis()
        if (now - lastCaptureTime < Config.captureCooldown * 1000L) {
            i( "Still in cooldown, skipping")
            return
        }
        lastCaptureTime = now
        captureCount++

        i( "Analyzing screen with VLM (mode=$backendMode)...")

        // Save captured image for verification (convert RGB to JPEG base64 for saving)
        val base64ForSave = rgbBytesToJpegBase64(rgbBytes, width, height)
        val imagePath = if (base64ForSave.isNotEmpty()) {
            saveCapturedImage(base64ForSave, captureCount)
        } else {
            null
        }

        iInference( "Running inference on RGB bytes...")

        if (backendMode == "LOCAL" && VlmApplication.localInferenceBackend != null) {
            // Use local inference from singleton - pass RGB bytes directly
            iInference( "Using LOCAL inference backend with RGB bytes")

            // Set up streaming callback
            VlmApplication.localInferenceBackend!!.onTokenStream = { token ->
                // Token streaming disabled - wait for full result
            }

            // Use the new analyzeRgbBytes method for direct RGB input
            val result = VlmApplication.localInferenceBackend!!.analyzeRgbBytes(
                rgbBytes = rgbBytes,
                width = width,
                height = height,
                prompt = Config.userPrompt,
                systemPrompt = Config.systemPrompt
            )

            iInference( "Local inference result: success=${result.success}, content length=${result.content.length}")

            if (result.success) {
                i( "Analysis complete: ${result.content.take(100)}...")
                i( "Tokens: prompt=${result.timings?.promptTokens} completion=${result.timings?.completionTokens}")
                i( "Total time: ${result.timings?.totalTimeMs}ms")
                // Calculate tokensPerSecond: (prompt + completion) / totalTimeMs (matches Camera mode)
                val promptTokens = result.timings?.promptTokens ?: 0
                val completionTokens = result.timings?.completionTokens ?: 0
                val totalTimeMs = result.timings?.totalTimeMs ?: 0L
                val tokensPerSecond = if (totalTimeMs > 0) {
                    ((promptTokens + completionTokens) * 1000.0) / totalTimeMs
                } else {
                    0.0
                }
                handler.post {
                    sendStreamComplete(result.content, promptTokens, completionTokens, totalTimeMs, imagePath, tokensPerSecond)
                }
            } else {
                Log.e(TAG, "API error: ${result.content}")
                handler.post {
                    sendError(result.content)
                }
            }
        } else {
            // Backend not initialized or wrong mode
            if (backendMode == "LOCAL") {
                Log.e(LOG_TAG_INFERENCE, "Local inference backend not initialized")
                handler.post {
                    sendError("Inference backend not initialized. Call initialize() first.")
                }
            } else {
                Log.e(LOG_TAG_INFERENCE, "RGB bytes mode only supported for LOCAL backend")
                handler.post {
                    sendError("RGB bytes mode only supported for LOCAL backend")
                }
            }
        }
    }

    /**
     * Convert RGB byte array to JPEG base64 for saving verification screenshot.
     */
    private fun rgbBytesToJpegBase64(rgbBytes: ByteArray, width: Int, height: Int): String {
        try {
            // Create bitmap from RGB bytes
            val pixels = IntArray(width * height)
            for (i in rgbBytes.indices step 3) {
                val pixelIndex = i / 3
                val r = rgbBytes[i].toInt() and 0xFF
                val g = rgbBytes[i + 1].toInt() and 0xFF
                val b = rgbBytes[i + 2].toInt() and 0xFF
                pixels[pixelIndex] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }

            val bitmap = android.graphics.Bitmap.createBitmap(pixels, width, height, android.graphics.Bitmap.Config.ARGB_8888)
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, Config.imageQuality, outputStream)
            bitmap.recycle()

            return android.util.Base64.encodeToString(outputStream.toByteArray(), android.util.Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert RGB bytes to JPEG", e)
            return ""
        }
    }

    private suspend fun analyzeScreen(base64Image: String) {
        iInference( "===== analyzeScreen START =====")
        iInference( "Backend mode: $backendMode")
        iInference( "VlmApplication.localInferenceBackend: ${VlmApplication.localInferenceBackend != null}")

        // Check cooldown
        val now = System.currentTimeMillis()
        if (now - lastCaptureTime < Config.captureCooldown * 1000L) {
            i( "Still in cooldown, skipping")
            return
        }
        lastCaptureTime = now
        captureCount++

        i( "Analyzing screen with VLM (mode=$backendMode)...")

        // Save captured image for verification
        val imagePath = saveCapturedImage(base64Image, captureCount)

        // Decode base64 to bytes for local inference
        val imageBytes = android.util.Base64.decode(base64Image, android.util.Base64.NO_WRAP)

        iInference( "Running inference...")

        if (backendMode == "LOCAL" && VlmApplication.localInferenceBackend != null) {
            // Use local inference from singleton with streaming
            iInference( "Using LOCAL inference backend from VlmApplication")

            // Set up streaming callback (no longer broadcasting tokens - just keep result at end)
            VlmApplication.localInferenceBackend!!.onTokenStream = { token ->
                // Token streaming disabled - wait for full result
            }

            val result = VlmApplication.localInferenceBackend!!.analyze(
                imageBytes = imageBytes,
                prompt = Config.userPrompt,
                systemPrompt = Config.systemPrompt
            )

            iInference( "Local inference result: success=${result.success}, content length=${result.content.length}")

            if (result.success) {
                i( "Analysis complete: ${result.content.take(100)}...")
                i( "Tokens: prompt=${result.timings?.promptTokens} completion=${result.timings?.completionTokens}")
                i( "Total time: ${result.timings?.totalTimeMs}ms")
                // Calculate tokensPerSecond: (prompt + completion) / totalTimeMs (matches Camera mode)
                val promptTokens = result.timings?.promptTokens ?: 0
                val completionTokens = result.timings?.completionTokens ?: 0
                val totalTimeMs = result.timings?.totalTimeMs ?: 0L
                val tokensPerSecond = if (totalTimeMs > 0) {
                    ((promptTokens + completionTokens) * 1000.0) / totalTimeMs
                } else {
                    0.0
                }
                handler.post {
                    sendStreamComplete(result.content, promptTokens, completionTokens, totalTimeMs, imagePath, tokensPerSecond)
                }
            } else {
                Log.e(TAG, "API error: ${result.content}")
                handler.post {
                    sendError(result.content)
                }
            }
        } else {
            // Use remote API
            iInference( "Using REMOTE API backend")
            val result = vlmAnalyzer.analyze(
                base64Image = base64Image,
                systemPrompt = Config.systemPrompt,
                userPrompt = Config.userPrompt,
                streamCallback = object : HttpClient.StreamCallback {
                    override fun onChunk(chunk: String, totalTokens: Int) {
                        // Token streaming disabled - wait for full result
                    }
                }
            )

            if (result.success) {
                i( "Analysis complete: ${result.content.take(100)}...")
                i( "Tokens: prompt=${result.promptTokens} completion=${result.completionTokens}")
                i( "Total time: ${result.totalTimeMs}ms")
                // Remote API doesn't provide tokensPerSecond - calculate from total tokens / time
                val tokensTotal = result.promptTokens + result.completionTokens
                val tokensPerSecond = if (result.totalTimeMs > 0) (tokensTotal * 1000.0 / result.totalTimeMs) else 0.0
                handler.post {
                    sendStreamComplete(result.content, result.promptTokens, result.completionTokens, result.totalTimeMs, imagePath, tokensPerSecond)
                }
            } else {
                Log.e(TAG, "API error: ${result.errorMessage}")
                handler.post {
                    sendError(result.errorMessage ?: "Unknown error")
                }
            }
        }
    }

    private fun saveCapturedImage(base64Image: String, captureNum: Int): String? {
        // Check if saving screenshots is enabled in config
        if (!Config.saveScreenshots) {
            d("Screenshot saving disabled in config, skipping")
            return null
        }

        try {
            val decodedBytes = android.util.Base64.decode(base64Image, android.util.Base64.NO_WRAP)
            val timestamp = System.currentTimeMillis()
            val fileName = "VLM_${timestamp}.jpg"  // JPEG format (matches rgbaToJpegBase64)

            // Save to Downloads/VLM_Capture/ folder using MediaStore (Android 10+ compatible)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = contentResolver
                val values = android.content.ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(MediaStore.Downloads.MIME_TYPE, "image/jpeg")  // JPEG MIME type
                    put(MediaStore.Downloads.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/VLM_Capture")
                }

                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    resolver.openOutputStream(uri)?.use { outputStream ->
                        outputStream.write(decodedBytes)
                        val path = "Downloads/VLM_Capture/$fileName"
                        i( "Saved captured image to: $path")
                        return path
                    }
                }
            }

            // Fallback: save to internal storage
            val capturesDir = File(filesDir, "captures")
            if (!capturesDir.exists()) {
                capturesDir.mkdirs()
            }

            val outputFile = File(capturesDir, fileName)
            outputFile.writeBytes(decodedBytes)
            val internalPath = "Internal: ${outputFile.absolutePath}"
            i( "Saved captured image to internal storage: $internalPath")
            return internalPath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save captured image", e)
            return null
        }
    }

    private fun sendStreamComplete(content: String, promptTokens: Int, completionTokens: Int, totalTimeMs: Long, imagePath: String?, tokensPerSecond: Double = 0.0) {
        i( "Stream complete: ${content.take(100)}...")

        // Send full result for display and TTS
        sendResult(content, promptTokens, completionTokens, totalTimeMs, imagePath, tokensPerSecond)
    }

    private fun sendResult(result: String, promptTokens: Int, completionTokens: Int, totalTimeMs: Long, imagePath: String?, tokensPerSecond: Double = 0.0) {
        i( "Analysis result: ${result.take(100)}...")

        val broadcast = Intent(ACTION_ANALYSIS_STREAM_COMPLETE).apply {
            putExtra(EXTRA_STREAM_CONTENT, result)
            putExtra(EXTRA_RESULT_PROMPT_TOKENS, promptTokens)
            putExtra(EXTRA_RESULT_COMPLETION_TOKENS, completionTokens)
            putExtra(EXTRA_RESULT_TOTAL_TIME_MS, totalTimeMs)
            putExtra(EXTRA_RESULT_TOKENS_PER_SECOND, tokensPerSecond)
            imagePath?.let { putExtra(EXTRA_IMAGE_PATH, it) }
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(broadcast)

        // Also broadcast capture saved event
        imagePath?.let { path ->
            val captureIntent = Intent(ACTION_CAPTURE_SAVED).apply {
                putExtra(EXTRA_IMAGE_PATH, path)
            }
            LocalBroadcastManager.getInstance(this).sendBroadcast(captureIntent)
        }
    }

    private fun sendError(error: String) {
        Log.e(TAG, "Analysis error: $error")

        val broadcast = Intent(ACTION_ANALYSIS_ERROR).apply {
            putExtra(EXTRA_ERROR_MESSAGE, error)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(broadcast)
    }
}
