// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

/**
 * CameraCaptureActivity - Camera preview and capture for VLM analysis.
 *
 * Features:
 * - 720p camera streaming at 30 FPS (YUV format, lightweight)
 * - Tap camera toggle button to switch between front/back camera
 * - Capture button or voice trigger to analyze current frame
 * - Floating overlay widget for controls and results
 * - Stream pauses during capture/analysis, resumes after
 *
 * Image Processing Flow (only on capture trigger):
 * 1. Capture latest YUV frame from ImageProxy
 * 2. Convert YUV to RGB byte array
 * 3. Apply scale factor (Config.imageScaleFactor)
 * 4. Send to inference backend (LOCAL: RGB bytes, REMOTE: JPEG base64)
 */
class CameraCaptureActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvStatus: TextView

    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private var preview: Preview? = null
    private var imageAnalysis: ImageAnalysis? = null

    // Latest captured frame data (copied RGB bytes - converted immediately)
    private var latestFrameRgbBytes: ByteArray? = null
    private var latestFrameWidth: Int = 0
    private var latestFrameHeight: Int = 0
    private var latestFrameTimestamp: Long = 0L  // Frame timestamp in nanoseconds
    private var latestFrameReceiveTimeMs: Long = 0L  // System time when frame was received
    private var frameCounter: Long = 0  // Count frames received

    // Streaming state
    private var isStreaming = false
    private var isCapturing = false  // True while processing capture

    // Backend mode (set when activity starts)
    private var backendMode: String = "REMOTE"  // "REMOTE" or "LOCAL"

    // Broadcast receiver state
    private var isReceiverRegistered = false

    private val handler = Handler(Looper.getMainLooper())

    companion object {
        private const val TAG = "VLM-Camera"
        private const val TARGET_RESOLUTION_WIDTH = 1440
        private const val TARGET_RESOLUTION_HEIGHT = 1080

        // Public log tags for logcat monitoring (consistent with ScreenCaptureService)
        const val LOG_TAG_CAMERA = "VLM-Camera"
        const val LOG_TAG_CAMERA_PREVIEW = "VLM-CameraPreview"
        const val LOG_TAG_CAMERA_CAPTURE = "VLM-CameraCapture"
        const val LOG_TAG_CAMERA_CONVERT = "VLM-CameraConvert"

        // Info log helper
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        // Debug log helper
        private fun d(msg: String) {
            if (VlmApplication.debugLogsEnabled) Log.d(TAG, msg)
        }

        // Broadcast actions
        const val ACTION_CAMERA_CAPTURE = "com.misar.vlmanalyze.CAMERA_CAPTURE"
        const val ACTION_CAMERA_STOP = "com.misar.vlmanalyze.CAMERA_STOP"
        const val ACTION_CAMERA_SWITCH = "com.misar.vlmanalyze.CAMERA_SWITCH"
        const val ACTION_CAMERA_RESOLUTION_READY = "com.misar.vlmanalyze.CAMERA_RESOLUTION_READY"
        const val EXTRA_BACKEND_MODE = "backend_mode"
        const val EXTRA_WIDTH = "width"
        const val EXTRA_HEIGHT = "height"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera_capture)

        i("CameraCaptureActivity created")

        // Get backend mode from intent
        backendMode = intent.getStringExtra(EXTRA_BACKEND_MODE) ?: "REMOTE"
        i("Backend mode: $backendMode")

        // Initialize UI
        previewView = findViewById(R.id.cameraPreview)
        tvStatus = findViewById(R.id.tv_camera_status)

        // Start camera stream
        startCameraStream()

        // Start overlay service
        OverlayViewService.startService(this)

        // Register broadcast receiver for capture triggers (voice)
        registerBroadcastReceiver()
    }

    override fun onDestroy() {
        super.onDestroy()
        i("CameraCaptureActivity destroying")

        // Stop camera stream
        stopCameraStream()

        // Stop overlay service
        OverlayViewService.stopService(this)

        // Unregister receiver (only if registered)
        if (isReceiverRegistered) {
            try {
                LocalBroadcastManager.getInstance(this).unregisterReceiver(broadcastReceiver)
                isReceiverRegistered = false
            } catch (e: Exception) {
                Log.e(TAG, "Failed to unregister receiver", e)
            }
        }

        // Clear any remaining frame data
        latestFrameRgbBytes = null
    }

    private fun startCameraStream() {
        i("Starting camera stream...")

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                // Create resolution selector for 4:3 aspect ratio (1440×1080 target, falls back to closest)
                // Both Preview and ImageAnalysis share the SAME resolution selector for consistency
                val resolutionSelector = ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            android.util.Size(TARGET_RESOLUTION_WIDTH, TARGET_RESOLUTION_HEIGHT),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()

                // Build preview use case with resolution selector
                preview = Preview.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setTargetRotation(android.view.Surface.ROTATION_0)  // Match display rotation
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // Build image analysis use case (YUV_420_888 format) with SAME resolution selector
                // Using minimum latency capture mode to avoid slow shutter speed (motion blur)
                imageAnalysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .setTargetRotation(android.view.Surface.ROTATION_0)  // Match preview rotation
                    .build()
                    .also {
                        it.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
                            // Log frame timestamp for debugging (in nanoseconds)
                            val timestamp = imageProxy.imageInfo.timestamp
                            val currentTime = System.currentTimeMillis()
                            d("=== FRAME ANALYZER ===")
                            d("Frame received: ${imageProxy.width}x${imageProxy.height}, timestamp: $timestamp ns")
                            d("Current time: $currentTime ms")
                            d("Analyzer executor: ${ContextCompat.getMainExecutor(this)}")
                            onFrameAvailable(imageProxy, timestamp)
                            d("=== END FRAME ANALYZER ===")
                        }
                    }

                // Bind use cases to camera
                bindCameraUseCases()

                isStreaming = true
                updateStatus("Camera streaming @ 720p")
                i("Camera stream started")

            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization failed", e)
                updateStatus("Camera error: ${e.message}")
                Toast.makeText(
                    this,
                    "Camera unavailable: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                handler.postDelayed({ finish() }, 2000)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        // Unbind all use cases before rebinding
        cameraProvider.unbindAll()

        try {
            // Bind use cases to camera
            val camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
            )

            d("Camera bound with selector: ${if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) "back" else "front"}")

        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
            updateStatus("Camera error: ${e.message}")
        }
    }

    private fun switchCamera() {
        i("Switching camera...")

        // Clear current frame data
        latestFrameRgbBytes = null

        // Toggle camera selector
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

        // Rebind use cases
        bindCameraUseCases()

        updateStatus("Camera switched")
        d("Camera switched to: ${if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) "back" else "front"}")
    }

    private fun onFrameAvailable(imageProxy: ImageProxy, frameTimestamp: Long = 0L) {
        val frameNum = frameCounter++
        val currentTime = System.currentTimeMillis()

        d("onFrameAvailable START: frame #$frameNum")
        d("  ImageProxy: ${imageProxy.width}x${imageProxy.height}")
        d("  Timestamp: $frameTimestamp ns")
        d("  Current time: $currentTime ms")

        // Convert to RGB bytes immediately using CameraX's toBitmap() - this is the proven working method
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val (rgbBytes, width, height) = CameraUtils.imageProxyToRgbBytes(imageProxy, rotationDegrees)

        // IMPORTANT: Close the ImageProxy to release it back to CameraX
        imageProxy.close()

        // Store the RGB bytes
        latestFrameRgbBytes = rgbBytes
        latestFrameWidth = width
        latestFrameHeight = height
        latestFrameTimestamp = frameTimestamp
        latestFrameReceiveTimeMs = currentTime

        d("  Frame converted to RGB: ${width}x${height}, ${rgbBytes.size} bytes")

        // Broadcast actual resolution on first frame
        if (frameCounter == 0L) {
            val intent = Intent(ACTION_CAMERA_RESOLUTION_READY).apply {
                putExtra(EXTRA_WIDTH, width)
                putExtra(EXTRA_HEIGHT, height)
            }
            LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
            d("Broadcast camera resolution: ${width}x${height}")
        }

        d("onFrameAvailable END: frame #$frameNum converted and stored")
    }

    private fun onCaptureTrigger() {
        if (isCapturing) {
            d("Capture already in progress, ignoring")
            return
        }

        i("Capture triggered")

        isCapturing = true

        // Pause camera stream (keep frame frozen during analysis)
        pauseCameraStream()

        // Show overlay immediately with "Analyzing..." status
        handler.post {
            // Broadcast to overlay to show "Analyzing..." status immediately
            val intent = Intent("com.misar.vlmanalyze.CAPTURE_STARTED")
            LocalBroadcastManager.getInstance(this@CameraCaptureActivity).sendBroadcast(intent)
        }

        // Capture and process frame
        handler.post {
            processCapturedFrame()
        }
    }

    private fun processCapturedFrame() {
        val rgbBytes = latestFrameRgbBytes
        if (rgbBytes == null) {
            d("No frame available for capture")
            handler.post {
                resumeCameraStream()
                updateStatus("No frame available")
                isCapturing = false
            }
            return
        }

        try {
            val captureTime = System.currentTimeMillis()
            val frameReceiveTime = latestFrameReceiveTimeMs
            val frameAgeMs = if (frameReceiveTime > 0) {
                (captureTime - frameReceiveTime)
            } else {
                0L
            }
            d("=== CAPTURE DIAGNOSTICS ===")
            d("Capturing frame: ${latestFrameWidth}x${latestFrameHeight}")
            d("Frame timestamp (ns): ${latestFrameTimestamp}")
            d("Frame age (from receive): ${frameAgeMs} ms")
            d("Capture time (ms): $captureTime")

            // Apply scale factor using stored dimensions
            val scaleStartTime = System.currentTimeMillis()
            val scaleFactor = Config.imageScaleFactorCamera
            val (scaledRgbBytes, scaledWidth, scaledHeight) = scaleRgbBytes(rgbBytes, latestFrameWidth, latestFrameHeight, scaleFactor)
            val scaleTime = System.currentTimeMillis() - scaleStartTime
            d("Scaling: ${scaleTime} ms, ${latestFrameWidth}x${latestFrameHeight} -> ${scaledWidth}x${scaledHeight}")
            d("=== END DIAGNOSTICS ===")

            // Clear the frame data
            latestFrameRgbBytes = null

            // Step 2.5: Save screenshot if enabled
            if (Config.saveScreenshots) {
                saveScreenshot(scaledRgbBytes, scaledWidth, scaledHeight)
            }

            // Step 3: Send to inference backend (run on IO thread to avoid UI freeze)
            kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.IO).launch {
                sendToInference(scaledRgbBytes, scaledWidth, scaledHeight)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing failed", e)
            handler.post {
                updateStatus("Error: ${e.message}")
                resumeCameraStream()
                isCapturing = false
            }
        }
    }

    private fun scaleRgbBytes(rgbBytes: ByteArray, width: Int, height: Int, scale: Float): Triple<ByteArray, Int, Int> {
        return CameraUtils.scaleRgbBytes(rgbBytes, width, height, scale)
    }

    private suspend fun sendToInference(rgbBytes: ByteArray, width: Int, height: Int) {
        d("Sending to $backendMode backend...")

        try {
            val result = if (backendMode == "LOCAL") {
                // LOCAL mode: use LocalInferenceBackend with RGB bytes
                // Use the singleton from VlmApplication if available
                val backend = VlmApplication.localInferenceBackend ?: LocalInferenceBackend(this).let {
                    val config = InferenceBackend.BackendConfig(
                        modelPath = VlmApplication.modelPath ?: "/data/local/tmp/llama.cpp/models/Qwen2-VL-2B-Q4_K_M.gguf",
                        mmprojPath = VlmApplication.mmprojPath ?: "/data/local/tmp/llama.cpp/models/mmproj-f16.gguf",
                        backend = VlmApplication.backendType,
                        maxTokens = Config.maxTokens,
                        temperature = Config.temperature.toFloat(),
                        topP = Config.topP.toFloat()
                    )
                    it.initialize(config)
                    VlmApplication.localInferenceBackend = it
                    it
                }
                backend.analyzeRgbBytes(rgbBytes, width, height, Config.userPrompt, Config.systemPrompt)
            } else {
                // REMOTE mode: convert RGB to JPEG base64 and use VlmAnalyzer
                val jpegBase64 = rgbBytesToJpegBase64(rgbBytes, width, height)
                val analyzer = VlmAnalyzer()
                analyzer.initialize(filesDir.absolutePath)
                analyzer.analyze(jpegBase64, Config.systemPrompt, Config.userPrompt)
            }

            d("Inference complete: ${when (result) {
                is InferenceBackend.InferenceResult -> result.success
                is HttpClient.ApiResult -> result.success
                else -> false
            }}")

            // Calculate tokens per second
            val totalTimeMs = if (result is InferenceBackend.InferenceResult) {
                result.timings?.totalTimeMs ?: 0L
            } else if (result is HttpClient.ApiResult) {
                result.totalTimeMs
            } else {
                0L
            }

            val tokensPerSecond = if (totalTimeMs > 0) {
                val totalTokens = when (result) {
                    is InferenceBackend.InferenceResult -> (result.timings?.promptTokens ?: 0) + (result.timings?.completionTokens ?: 0)
                    is HttpClient.ApiResult -> result.promptTokens + result.completionTokens
                    else -> 0
                }
                (totalTokens * 1000.0) / totalTimeMs
            } else {
                0.0
            }

            // Broadcast result to overlay for display and TTS (via LocalBroadcastManager)
            val intent = Intent(ScreenCaptureService.ACTION_ANALYSIS_STREAM_COMPLETE).apply {
                putExtra(ScreenCaptureService.EXTRA_STREAM_CONTENT, when (result) {
                    is InferenceBackend.InferenceResult -> result.content
                    is HttpClient.ApiResult -> result.content
                    else -> ""
                })
                putExtra(ScreenCaptureService.EXTRA_RESULT_PROMPT_TOKENS, when (result) {
                    is InferenceBackend.InferenceResult -> result.timings?.promptTokens ?: 0
                    is HttpClient.ApiResult -> result.promptTokens
                    else -> 0
                })
                putExtra(ScreenCaptureService.EXTRA_RESULT_COMPLETION_TOKENS, when (result) {
                    is InferenceBackend.InferenceResult -> result.timings?.completionTokens ?: 0
                    is HttpClient.ApiResult -> result.completionTokens
                    else -> 0
                })
                putExtra(ScreenCaptureService.EXTRA_RESULT_TOTAL_TIME_MS, totalTimeMs)
                putExtra(ScreenCaptureService.EXTRA_RESULT_TOKENS_PER_SECOND, tokensPerSecond)
            }
            LocalBroadcastManager.getInstance(this@CameraCaptureActivity).sendBroadcast(intent)

        } catch (e: Exception) {
            d("Inference failed: ${e.message}")
            // Broadcast error to overlay (via LocalBroadcastManager)
            val intent = Intent(ScreenCaptureService.ACTION_ANALYSIS_ERROR).apply {
                putExtra(ScreenCaptureService.EXTRA_ERROR_MESSAGE, e.message ?: "Unknown error")
            }
            LocalBroadcastManager.getInstance(this@CameraCaptureActivity).sendBroadcast(intent)
        } finally {
            handler.post {
                // Don't resume camera yet - wait for TTS to complete
                // Camera stays frozen while results are being spoken
                isCapturing = false
            }
        }
    }

    private fun rgbBytesToJpegBase64(rgbBytes: ByteArray, width: Int, height: Int): String {
        return CameraUtils.rgbBytesToJpegBase64(rgbBytes, width, height, Config.imageQuality)
    }

    private fun saveScreenshot(rgbBytes: ByteArray, width: Int, height: Int) {
        try {
            // Convert RGB bytes to bitmap
            val pixels = IntArray(width * height)
            var i = 0
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val r = rgbBytes[i++].toInt() and 0xFF
                    val g = rgbBytes[i++].toInt() and 0xFF
                    val b = rgbBytes[i++].toInt() and 0xFF
                    pixels[y * width + x] = -0x1000000 or (r shl 16) or (g shl 8) or b
                }
            }

            val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
            val timestamp = System.currentTimeMillis()
            val fileName = "VLM_CAMERA_${timestamp}.png"

            // Save to Downloads/VLM_Capture/ folder using MediaStore (Android 10+ compatible)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = contentResolver
                val values = android.content.ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(MediaStore.Downloads.MIME_TYPE, "image/png")
                    put(MediaStore.Downloads.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/VLM_Capture")
                }

                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                uri?.let {
                    resolver.openOutputStream(it)?.use { outputStream ->
                        // PNG is lossless - no quality parameter needed
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
                        i("Saved camera screenshot (PNG) to: Downloads/VLM_Capture/$fileName")
                    }
                }
            }
            bitmap.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save camera screenshot", e)
        }
    }

    private fun pauseCameraStream() {
        d("Pausing camera stream")
        cameraProvider?.unbindAll()
        isStreaming = false
    }

    private fun resumeCameraStream() {
        d("Resuming camera stream")
        bindCameraUseCases()
        isStreaming = true
    }

    private fun stopCameraStream() {
        d("Stopping camera stream")
        cameraProvider?.unbindAll()
        isStreaming = false
        // Clear the latest frame data
        latestFrameRgbBytes = null
    }

    private fun updateStatus(status: String) {
        handler.post {
            tvStatus.text = status
        }
    }

    private fun registerBroadcastReceiver() {
        // Register for LocalBroadcastManager (receives triggers from OverlayViewService)
        val filter = IntentFilter().apply {
            addAction(ScreenCaptureService.ACTION_TRIGGER_CAPTURE)  // From overlay/voice
            addAction(CameraCaptureActivity.ACTION_CAMERA_CAPTURE)  // Internal action
            addAction(CameraCaptureActivity.ACTION_CAMERA_STOP)
            addAction(CameraCaptureActivity.ACTION_CAMERA_SWITCH)   // Camera switch from overlay
            addAction("com.misar.vlmanalyze.TTS_COMPLETE")          // TTS complete from overlay
        }
        LocalBroadcastManager.getInstance(this).registerReceiver(broadcastReceiver, filter)
        i("LocalBroadcastManager receiver registered for actions: TRIGGER_CAPTURE, CAMERA_CAPTURE, CAMERA_STOP, CAMERA_SWITCH, TTS_COMPLETE")
        isReceiverRegistered = true
    }

    private val broadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            i("Broadcast received with action: ${intent?.action}")
            when (intent?.action) {
                ACTION_CAMERA_CAPTURE, ScreenCaptureService.ACTION_TRIGGER_CAPTURE -> {
                    i("Capture trigger received in CameraCaptureActivity")
                    d("Capture trigger received")
                    onCaptureTrigger()
                }
                ACTION_CAMERA_STOP -> {
                    d("Stop command received")
                    finish()
                }
                ACTION_CAMERA_SWITCH -> {
                    i("Camera switch command received")
                    switchCamera()
                }
                "com.misar.vlmanalyze.TTS_COMPLETE" -> {
                    i("TTS complete - resuming camera stream")
                    handler.post {
                        resumeCameraStream()
                        updateStatus("Camera ready - Use overlay to capture")
                    }
                }
            }
        }
    }
}
