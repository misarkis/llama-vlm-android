// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.content.res.ColorStateList
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import android.widget.RadioGroup
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity(), TtsHelper.Listener {

    private lateinit var mediaProjectionManager: MediaProjectionManager
    private val ttsHelper = TtsHelper(this)
    private val audioRecorder = AudioRecorder()
    private val handler = Handler(Looper.getMainLooper())
    private var isServiceRunning = false
    private var isOverlayEnabled = false
    private var broadcastReceiver: android.content.BroadcastReceiver? = null
    private var localBackend: com.misar.vlmanalyze.LocalBackend? = null
    private var currentMode: BackendMode = BackendMode.REMOTE

    // Capture source selection (Screen vs Camera)
    private var captureSource: CaptureSource = CaptureSource.SCREEN

    enum class CaptureSource {
        SCREEN,
        CAMERA
    }

    // Camera resolution (discovered from camera and persisted)
    private var cameraWidth = 0
    private var cameraHeight = 0

    // UI references for image scale preview (initialized in onCreate)
    private lateinit var imageScaleSlider: android.widget.SeekBar
    private lateinit var imageScaleValueText: TextView
    private lateinit var imageSizePreviewText: TextView

    // Screen dimensions (initialized in onCreate)
    private var screenWidth = 0
    private var screenHeight = 0

    // Selected model paths (full paths)
    private var selectedModelPath: String? = null
    private var selectedMmprojPath: String? = null

    enum class BackendMode {
        REMOTE,
        LOCAL
    }

    // Backend selection options for LOCAL mode
    enum class LocalBackendType(val displayName: String) {
        CPU("CPU - ARM Cores"),
        GPU("GPU - Adreno"),
        NPU("NPU - Hexagon HTP")
    }

    companion object {
        private const val TAG = "VLM-MainActivity"
        const val LOG_TAG_MAIN = "VLM-MainActivity"

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }
        private const val SCREEN_CAPTURE_REQUEST_CODE = 100
        private const val CAMERA_CAPTURE_REQUEST_CODE = 101
        private const val MICROPHONE_PERMISSION_REQUEST_CODE = 200
        private const val OVERLAY_PERMISSION_REQUEST_CODE = 300
        private const val PREFS_NAME = "VLM_Preferences"
        private const val KEY_RESULT_CODE = "media_projection_result_code"
        private const val KEY_DATA_URI = "media_projection_data_uri"
        private const val KEY_MODE = "backend_mode"
        private const val KEY_LOCAL_BACKEND = "local_backend"
        private const val KEY_MODEL_PATH = "model_path"
        private const val KEY_MMPROJ_PATH = "mmproj_path"
        private const val KEY_SERVER_N_CTX = "server_n_ctx"
        private const val KEY_SERVER_N_BATCH = "server_n_batch"
        private const val KEY_SERVER_VERBOSE = "server_verbose"
        private const val KEY_DEBUG_LOGS_ENABLED = "debug_logs_enabled"
        private const val KEY_INFO_LOGS_ENABLED = "info_logs_enabled"
        private const val KEY_HTP_SESSIONS = "htp_sessions"
        private const val KEY_CAPTURE_SOURCE = "capture_source"
        // Camera resolution persistence
        private const val KEY_CAMERA_WIDTH = "camera_width"
        private const val KEY_CAMERA_HEIGHT = "camera_height"
        // Inference parameters
        private const val KEY_IMAGE_MIN_TOKENS = "image_min_tokens"
        private const val KEY_TOP_K = "top_k"
        private const val KEY_TOP_P = "top_p"
        private const val KEY_TEMPERATURE = "temperature"
        private const val KEY_REPETITION_PENALTY = "repetition_penalty"
        private const val REQUEST_MODEL_PICKER = 1001
        private const val REQUEST_MMPROJ_PICKER = 1002

        // Models directory on device - using /sdcard/Download for accessibility
        // Models directory on device
        const val MODELS_DIR = "/data/local/tmp/llama.cpp/models"

        // Storage access framework URI for persisted permissions
        var persistedModelsUri: android.net.Uri? = null

        private val DATA_EXTRAS_KEYS = setOf(
            "android.intent.extra.result.RESULT_CODE",
            "media_projection",
            "android.media.projection"
        )
    }

    init {
        System.loadLibrary("vlmanalyze")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        i( "MainActivity created")

        mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        nativeInit(assets, filesDir.absolutePath)
        ttsHelper.initialize(this)

        // Initialize LocalBackend (for HTTP server approach)
        localBackend = LocalBackend(this)

        // LocalInferenceBackend is initialized lazily on first use

        setContentView(R.layout.activity_main)

        val startBtn = findViewById<android.widget.Button>(R.id.btn_start)
        val stopBtn = findViewById<android.widget.Button>(R.id.btn_stop)
        val settingsBtn = findViewById<android.widget.Button>(R.id.btn_settings)
        val btnModeRemote = findViewById<android.widget.Button>(R.id.btn_mode_remote)
        val saveAudioCheck = findViewById<android.widget.CheckBox>(R.id.cb_save_audio)
        val saveScreenshotsCheck = findViewById<android.widget.CheckBox>(R.id.cb_save_screenshots)
        val btnModeLocal = findViewById<android.widget.Button>(R.id.btn_mode_local)
        val localServerSection = findViewById<androidx.cardview.widget.CardView>(R.id.card_local_server)
        val btnStartServer = findViewById<android.widget.Button>(R.id.btn_start_server)
        val btnStopServer = findViewById<android.widget.Button>(R.id.btn_stop_server)
        val serverStatusText = findViewById<android.widget.TextView>(R.id.tv_server_status)
        val rbScreenCapture = findViewById<android.widget.RadioButton>(R.id.rb_screen_capture)
        val rbCameraCapture = findViewById<android.widget.RadioButton>(R.id.rb_camera_capture)

        // Load saved config
        loadLocalBackendConfig()
        loadCaptureSource()
        loadCameraResolution()

        // Load server config (includes debug/info logs settings) and apply to VlmApplication
        val serverConfig = loadServerConfig()
        VlmApplication.debugLogsEnabled = serverConfig.debugLogsEnabled
        VlmApplication.infoLogsEnabled = serverConfig.infoLogsEnabled
        nativeSetDebugLogsEnabled(serverConfig.debugLogsEnabled)
        nativeSetInfoLogsEnabled(serverConfig.infoLogsEnabled)

        // Update local server section visibility based on loaded mode
        if (currentMode == BackendMode.LOCAL) {
            localServerSection.visibility = android.view.View.VISIBLE
        } else {
            localServerSection.visibility = android.view.View.GONE
        }

        // Update capture source UI
        updateCaptureSourceUI()

        updateModeButtons()
        updateSettingsButtonVisibility()

        // Initialize save options checkboxes
        saveAudioCheck.isChecked = Config.saveAudio
        saveScreenshotsCheck.isChecked = Config.saveScreenshots
        saveAudioCheck.setOnCheckedChangeListener { _, isChecked ->
            Config.saveAudio = isChecked
            Config.save(filesDir.absolutePath)
        }
        saveScreenshotsCheck.setOnCheckedChangeListener { _, isChecked ->
            Config.saveScreenshots = isChecked
            Config.save(filesDir.absolutePath)
        }

        // Capture source toggle - use RadioGroup for mutual exclusion
        val radioGroup = findViewById<RadioGroup>(R.id.rg_capture_source)

        radioGroup.setOnCheckedChangeListener { group, checkedId ->
            when (checkedId) {
                R.id.rb_screen_capture -> {
                    captureSource = CaptureSource.SCREEN
                    saveCaptureSource()
                    updateStatus("Screen capture selected")
                    // Update slider to show screen's scale factor
                    val screenScale = Config.imageScaleFactorScreen
                    imageScaleSlider.progress = ((screenScale * 100).toInt()).coerceIn(10, 100)
                    imageScaleValueText.text = String.format("%.2f", screenScale)
                    updateImageSizePreview(screenScale)
                }
                R.id.rb_camera_capture -> {
                    captureSource = CaptureSource.CAMERA
                    saveCaptureSource()
                    updateStatus("Camera capture selected")
                    // Update slider to show camera's scale factor
                    val cameraScale = Config.imageScaleFactorCamera
                    imageScaleSlider.progress = ((cameraScale * 100).toInt()).coerceIn(10, 100)
                    imageScaleValueText.text = String.format("%.2f", cameraScale)
                    updateImageSizePreview(cameraScale)
                }
            }
        }

        // Mode switch buttons
        btnModeRemote.setOnClickListener {
            currentMode = BackendMode.REMOTE
            localServerSection.visibility = android.view.View.GONE
            findViewById<android.widget.Button>(R.id.btn_server_settings).visibility = android.view.View.GONE
            findViewById<android.widget.Button>(R.id.btn_settings).visibility = android.view.View.VISIBLE
            saveMode()
            updateModeButtons()
            updateStatus("REMOTE mode - Using cloud API")
        }

        btnModeLocal.setOnClickListener {
            currentMode = BackendMode.LOCAL
            localServerSection.visibility = android.view.View.VISIBLE
            findViewById<android.widget.Button>(R.id.btn_server_settings).visibility = android.view.View.VISIBLE
            findViewById<android.widget.Button>(R.id.btn_settings).visibility = android.view.View.GONE
            saveMode()
            updateModeButtons()
            updateStatus("LOCAL mode - Configure and start server")
        }

        // Image Scale Factor slider - shared for both modes
        imageScaleSlider = findViewById(R.id.sb_image_scale_factor)
        imageScaleValueText = findViewById(R.id.tv_image_scale_value)
        imageSizePreviewText = findViewById(R.id.tv_image_size_preview)

        // Get actual screen dimensions
        val displayMetrics = resources.displayMetrics
        screenWidth = displayMetrics.widthPixels
        screenHeight = displayMetrics.heightPixels

        // Initialize slider value based on current capture source
        val initialScale = if (captureSource == CaptureSource.CAMERA) Config.imageScaleFactorCamera else Config.imageScaleFactorScreen
        val sliderProgress = ((initialScale * 100).toInt()).coerceIn(10, 100)
        imageScaleSlider.progress = sliderProgress
        imageScaleValueText.text = String.format("%.2f", initialScale)

        // Update image size preview based on capture source
        updateImageSizePreview(initialScale)

        // Update on slider change
        imageScaleSlider.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                val value = (progress / 100f).coerceIn(0.1f, 1.0f)
                imageScaleValueText.text = String.format("%.2f", value)
                updateImageSizePreview(value)
                // Update the appropriate config value in real-time
                when (captureSource) {
                    CaptureSource.CAMERA -> Config.imageScaleFactorCamera = value
                    CaptureSource.SCREEN -> Config.imageScaleFactorScreen = value
                }
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                // Save on stop tracking - save to correct config based on capture source
                val value = imageScaleSlider.progress / 100f
                when (captureSource) {
                    CaptureSource.CAMERA -> Config.imageScaleFactorCamera = value
                    CaptureSource.SCREEN -> Config.imageScaleFactorScreen = value
                }
                Config.save(filesDir.absolutePath)
            }
        })

        // "Start Analyzer" button - starts both overlay and screen/camera capture
        startBtn.setOnClickListener {
            // First check overlay permission
            val hasOverlayPermission = Settings.canDrawOverlays(this)
            if (!hasOverlayPermission) {
                i( "Requesting overlay permission first")
                val intent = Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:$packageName")
                )
                startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE)
                return@setOnClickListener
            }

            // Permission granted - proceed with starting analyzer

            // Handle based on capture source
            when (captureSource) {
                CaptureSource.CAMERA -> {
                    // Request camera permission first
                    if (CameraPermissionHelper.hasCameraPermission(this)) {
                        // Permission granted - launch camera activity
                        i( "Launching camera capture")
                        val intent = Intent(this, CameraCaptureActivity::class.java).apply {
                            putExtra(CameraCaptureActivity.EXTRA_BACKEND_MODE, if (currentMode == BackendMode.LOCAL) "LOCAL" else "REMOTE")
                        }
                        startActivity(intent)
                        updateStatus("Camera mode - Tap capture button to analyze")
                    } else {
                        // Request permission
                        CameraPermissionHelper.requestCameraPermission(this)
                        // Store current mode to resume after permission granted
                        VlmApplication.pendingCaptureSource = captureSource
                        VlmApplication.pendingBackendMode = if (currentMode == BackendMode.LOCAL) "LOCAL" else "REMOTE"
                        updateStatus("Requesting camera permission...")
                    }
                    return@setOnClickListener
                }

                CaptureSource.SCREEN -> {
                    // Original screen capture flow
                    if (ScreenCaptureService.isServiceRunning()) {
                        // Service is running - just trigger a capture
                        i( "Triggering capture")
                        ScreenCaptureService.triggerCapture(this)
                        updateStatus("🔍 Analyzing Screen...")
                        return@setOnClickListener
                    }

                    // Start overlay first
                    try {
                        i( "Starting OverlayViewService")
                        OverlayViewService.startService(this)
                        isOverlayEnabled = true
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to start overlay", e)
                        updateStatus("Error starting overlay: ${e.message}")
                        Toast.makeText(this, "Failed to start overlay", Toast.LENGTH_SHORT).show()
                        return@setOnClickListener
                    }

                    // Then check for persisted credentials and start screen capture
                    val persistedData = getPersistedMediaProjectionData()
                    if (persistedData != null) {
                        i( "Using persisted media projection credentials")
                        isServiceRunning = true
                        VlmApplication.pendingResult = persistedData.first
                        VlmApplication.pendingData = persistedData.second

                        // Create Intent with media projection data and backend mode
                        val intent = Intent(this, ScreenCaptureService::class.java)
                        intent.putExtra("resultCode", persistedData.first)
                        intent.putExtra("media_projection_data", persistedData.second)
                        intent.putExtra(ScreenCaptureService.EXTRA_BACKEND_MODE, if (currentMode == BackendMode.LOCAL) "LOCAL" else "REMOTE")

                        startForegroundService(intent)
                        updateStatus("Service running - Tap 'Capture' on overlay to analyze")
                        Toast.makeText(this, "Analyzer started!", Toast.LENGTH_SHORT).show()
                    } else {
                        // Service not running - start the screen capture permission flow
                        i( "Starting screen capture permission flow")
                        startCaptureRequest()
                    }
                }
            }
        }

        stopBtn.setOnClickListener {
            stopCapture()
        }

        // Settings button - show config dialog
        settingsBtn.setOnClickListener {
            showSettingsDialog()
        }

        // Setup local server controls
        setupLocalServerControls(
            btnStartServer = btnStartServer,
            btnStopServer = btnStopServer,
            serverStatusText = serverStatusText
        )

        registerBroadcastReceiver()
        requestMicrophonePermission()
        checkStoragePermission()
        updateStatus("Ready - Tap 'Analyze Screen' to start")
    }

    private fun checkStoragePermission() {
        // Check if models directory exists and is readable
        val modelsDir = java.io.File(MODELS_DIR)
        i( "Models dir: exists=${modelsDir.exists()}, canRead=${modelsDir.canRead()}, isDir=${modelsDir.isDirectory}")

        if (modelsDir.exists() && modelsDir.isDirectory) {
            val files = modelsDir.listFiles()
            i( "Models dir contains ${files?.size ?: 0} entries")
            files?.forEach { f ->
                i( "  - ${f.name} (dir=${f.isDirectory})")
            }
        } else {
            Log.w(TAG, "Models directory not accessible: $MODELS_DIR")
        }
    }

    private fun copyFileToPrivateStorage(sourcePath: String, isModel: Boolean): String {
        val sourceFile = File(sourcePath)
        if (!sourceFile.exists()) {
            Log.e(TAG, "Source file not found: $sourcePath")
            return sourcePath
        }

        // Delete only the old file of the same type (model deletes model, mmproj deletes mmproj)
        if (isModel) {
            val oldModelName = selectedModelPath?.let { File(it).name }
            if (oldModelName != null) {
                val oldModel = File(filesDir, oldModelName)
                if (oldModel.exists()) {
                    oldModel.delete()
                    i( "Deleted old model: ${oldModel.absolutePath}")
                }
            }
        } else {
            val oldMmprojName = selectedMmprojPath?.let { File(it).name }
            if (oldMmprojName != null) {
                val oldMmproj = File(filesDir, oldMmprojName)
                if (oldMmproj.exists()) {
                    oldMmproj.delete()
                    i( "Deleted old mmproj: ${oldMmproj.absolutePath}")
                }
            }
        }

        // Copy to app's private files directory
        val destFile = File(filesDir, sourceFile.name)
        try {
            sourceFile.copyTo(destFile, overwrite = true)
            i( "Copied ${sourceFile.name} to private storage: ${destFile.absolutePath}")
            return destFile.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy file: ${e.message}")
            return sourcePath
        }
    }

    private fun requestStorageAccess() {
        // Use Storage Access Framework to get access to external storage
        val intent = android.content.Intent(android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
        try {
            startActivity(intent)
        } catch (e: Exception) {
            Log.w(TAG, "Could not open file access settings", e)
        }
    }

    private fun requestMicrophonePermission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.RECORD_AUDIO),
                MICROPHONE_PERMISSION_REQUEST_CODE
            )
        } else {
            i( "Microphone permission already granted")
        }
    }

    private fun checkOverlayPermissionAndEnableStartButton() {
        val hasOverlayPermission = Settings.canDrawOverlays(this)
        val startBtn = findViewById<android.widget.Button>(R.id.btn_start)

        if (!hasOverlayPermission) {
            // Show message but keep button enabled - tapping will take to permission screen
            startBtn.text = "Start Analyzer"
            startBtn.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#4CAF50"))
            updateStatus("Overlay permission required - tap 'Start Analyzer' to enable")
        } else {
            // Permission already granted
            startBtn.text = "Start Analyzer"
            startBtn.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#4CAF50"))
            updateStatus("Ready - Tap 'Start Analyzer' to begin")
        }
    }

    private fun startCaptureRequest() {
        i( "Requesting screen capture permission")
        val intent = mediaProjectionManager.createScreenCaptureIntent()
        startActivityForResult(intent, SCREEN_CAPTURE_REQUEST_CODE)
    }

    private fun getSharedPreferences(): SharedPreferences {
        return getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    private fun persistMediaProjectionData(resultCode: Int, data: Intent) {
        val prefs = getSharedPreferences()
        val editor = prefs.edit()
        editor.putInt(KEY_RESULT_CODE, resultCode)

        // Store URI data if available
        data.data?.let { uri ->
            editor.putString(KEY_DATA_URI, uri.toString())
        }

        // Store extras (key-value pairs from the Intent)
        data.extras?.keySet()?.forEach { key ->
            when (val value = data.extras?.get(key)) {
                is String -> editor.putString(key, value)
                is Int -> editor.putInt(key, value)
                is Long -> editor.putLong(key, value)
                is Float -> editor.putFloat(key, value)
                is Boolean -> editor.putBoolean(key, value)
            }
        }

        editor.apply()
        i( "Persisted media projection credentials")
    }

    private fun getPersistedMediaProjectionData(): Pair<Int, Intent>? {
        val prefs = getSharedPreferences()
        val resultCode = prefs.getInt(KEY_RESULT_CODE, -1)

        if (resultCode == -1) {
            return null
        }

        // Reconstruct Intent from stored data
        val intent = Intent()

        // Restore URI if available
        prefs.getString(KEY_DATA_URI, null)?.let { uriString ->
            try {
                intent.data = Uri.parse(uriString)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to restore URI: ${e.message}")
            }
        }

        // Restore extras - restore all stored keys except our internal ones
        prefs.all.keys.filterNot { it == KEY_RESULT_CODE || it == KEY_DATA_URI }.forEach { key ->
            when (val value = prefs.all[key]) {
                is String -> intent.putExtra(key, value)
                is Int -> intent.putExtra(key, value)
                is Long -> intent.putExtra(key, value)
                is Float -> intent.putExtra(key, value)
                is Boolean -> intent.putExtra(key, value)
            }
        }

        i( "Restored persisted media projection credentials")
        return Pair(resultCode, intent)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            MICROPHONE_PERMISSION_REQUEST_CODE -> {
                if (permissions.contains(android.Manifest.permission.RECORD_AUDIO)) {
                    if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                        i( "Microphone permission granted")
                    } else {
                        Toast.makeText(this, "Voice input disabled", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            CAMERA_CAPTURE_REQUEST_CODE -> {
                // Handle camera permission result
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Permission granted - start camera activity
                    i( "Camera permission granted")
                    val backendMode = VlmApplication.pendingBackendMode ?: if (currentMode == BackendMode.LOCAL) "LOCAL" else "REMOTE"
                    val intent = Intent(this, CameraCaptureActivity::class.java).apply {
                        putExtra(CameraCaptureActivity.EXTRA_BACKEND_MODE, backendMode)
                    }
                    startActivity(intent)
                    updateStatus("Camera mode - Tap capture button to analyze")
                    VlmApplication.pendingCaptureSource = null
                    VlmApplication.pendingBackendMode = null
                } else {
                    Toast.makeText(this, "Camera permission denied - cannot use camera mode", Toast.LENGTH_LONG).show()
                    updateStatus("Camera permission denied")
                    VlmApplication.pendingCaptureSource = null
                    VlmApplication.pendingBackendMode = null
                }
            }
        }
    }

    // Removed - screen capture now starts immediately without notification permission delay

    private fun stopCapture() {
        i( "Stopping service")
        isServiceRunning = false
        ScreenCaptureService.stopService(this)
        // Disable overlay when stopping
        if (isOverlayEnabled) {
            OverlayViewService.stopService(this)
            isOverlayEnabled = false
        }
        // Clear persisted credentials when user explicitly stops
        clearPersistedMediaProjectionData()
        checkOverlayPermissionAndEnableStartButton()
        updateStatus("Stopped")
    }

    private fun clearPersistedMediaProjectionData() {
        val prefs = getSharedPreferences()
        prefs.edit()
            .remove(KEY_RESULT_CODE)
            .remove(KEY_DATA_URI)
            .apply()
        i( "Cleared persisted media projection credentials")
    }

    private fun updateStatus(status: String) {
        handler.post {
            findViewById<android.widget.TextView>(R.id.status_text)?.text = status
        }
    }

    private fun saveCaptureSource() {
        val prefs = getSharedPreferences()
        prefs.edit().putString(KEY_CAPTURE_SOURCE, captureSource.name).apply()
        i("Saved capture source: ${captureSource.name}")
    }

    private fun loadCaptureSource() {
        val prefs = getSharedPreferences()
        val source = prefs.getString(KEY_CAPTURE_SOURCE, "SCREEN")
        captureSource = if (source == "CAMERA") CaptureSource.CAMERA else CaptureSource.SCREEN
        i("Loaded capture source: ${captureSource.name}")
    }

    private fun loadCameraResolution() {
        val prefs = getSharedPreferences()
        cameraWidth = prefs.getInt(KEY_CAMERA_WIDTH, 0)
        cameraHeight = prefs.getInt(KEY_CAMERA_HEIGHT, 0)
        i("Loaded camera resolution: ${cameraWidth}x${cameraHeight}")
    }

    private fun saveCameraResolution(width: Int, height: Int) {
        val prefs = getSharedPreferences()
        prefs.edit()
            .putInt(KEY_CAMERA_WIDTH, width)
            .putInt(KEY_CAMERA_HEIGHT, height)
            .apply()
        i("Saved camera resolution: ${width}x${height}")
    }

    private fun updateCaptureSourceUI() {
        val rbScreenCapture = findViewById<android.widget.RadioButton>(R.id.rb_screen_capture)
        val rbCameraCapture = findViewById<android.widget.RadioButton>(R.id.rb_camera_capture)
        rbScreenCapture.isChecked = captureSource == CaptureSource.SCREEN
        rbCameraCapture.isChecked = captureSource == CaptureSource.CAMERA
    }

    // JNI-compatible signature (no default params) - must match exactly what native code expects
    fun onAnalysisResult(result: String, promptTokens: Int, completionTokens: Int, totalTimeMs: Long, tokensPerSecond: Double, imagePath: String?) {
        handler.post {
            val statsText = if (totalTimeMs > 0) {
                val tokensTotal = promptTokens + completionTokens
                val tps = if (tokensPerSecond > 0) tokensPerSecond.toString().take(5) else "0"
                "\n\n---\nProcessing: ${totalTimeMs}ms | Tokens: ${tokensTotal} (prompt=${promptTokens}, completion=${completionTokens}) | ${tps} t/s"
            } else {
                ""
            }
            val imageText = imagePath?.let { "\n\nImage saved: $it" } ?: ""
            findViewById<android.widget.TextView>(R.id.result_text)?.text = result + statsText + imageText
            val tokensTotal = promptTokens + completionTokens
            val tps = if (tokensPerSecond > 0) tokensPerSecond.toString().take(4) + " t/s" else ""
            updateStatus(if (totalTimeMs > 0) "Analysis complete | ${tokensTotal} tokens ${tps}" else "Analysis complete")

            // Broadcast result to overlay service for display and TTS
            val intent = Intent(this, OverlayViewService::class.java).apply {
                action = ScreenCaptureService.ACTION_ANALYSIS_STREAM_COMPLETE
                putExtra(ScreenCaptureService.EXTRA_STREAM_CONTENT, result)
                putExtra(ScreenCaptureService.EXTRA_RESULT_PROMPT_TOKENS, promptTokens)
                putExtra(ScreenCaptureService.EXTRA_RESULT_COMPLETION_TOKENS, completionTokens)
                putExtra(ScreenCaptureService.EXTRA_RESULT_TOTAL_TIME_MS, totalTimeMs)
                putExtra(ScreenCaptureService.EXTRA_RESULT_TOKENS_PER_SECOND, tokensPerSecond)
            }
            startService(intent)
        }
    }

    fun onAnalysisError(error: String) {
        handler.post {
            findViewById<android.widget.TextView>(R.id.result_text)?.text = "Error: $error"
            updateStatus("Error")
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        i( "onActivityResult: requestCode=$requestCode, resultCode=$resultCode")

        if (requestCode == OVERLAY_PERMISSION_REQUEST_CODE) {
            if (Settings.canDrawOverlays(this)) {
                i( "Overlay permission granted")
                checkOverlayPermissionAndEnableStartButton()
                Toast.makeText(this, "Permission granted - tap 'Start Analyzer' to begin", Toast.LENGTH_LONG).show()
            } else {
                Log.w(TAG, "Overlay permission denied")
                checkOverlayPermissionAndEnableStartButton()
                Toast.makeText(this, "Overlay permission required to use analyzer", Toast.LENGTH_LONG).show()
            }
        } else if (requestCode == SCREEN_CAPTURE_REQUEST_CODE) {
            i( "Screen capture result: resultCode=$resultCode, data=${data != null}")
            if (resultCode == RESULT_OK && data != null) {
                i( "Screen capture permission GRANTED - storing in singleton and starting service")
                isServiceRunning = true
                // Store in singleton as primary source (more reliable than Intent extras)
                VlmApplication.pendingResult = resultCode
                VlmApplication.pendingData = data
                i( "Stored in singleton: resultCode=$resultCode, data!=null=${data != null}")
                // Persist credentials for future use
                persistMediaProjectionData(resultCode, data)

                // Create Intent with media projection data and backend mode
                val intent = Intent(this, ScreenCaptureService::class.java)
                intent.putExtra("resultCode", resultCode)
                intent.putExtra("media_projection_data", data)
                intent.putExtra(ScreenCaptureService.EXTRA_BACKEND_MODE, if (currentMode == BackendMode.LOCAL) "LOCAL" else "REMOTE")

                startForegroundService(intent)
                updateStatus("Service running - Tap 'Capture' on overlay to analyze")
                Toast.makeText(this, "Screen capture started!", Toast.LENGTH_SHORT).show()
            } else {
                Log.w(TAG, "Screen capture permission DENIED")
                updateStatus("Screen capture permission denied")
                Toast.makeText(this, "Screen capture denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onSpeakingComplete() {}
    
    private fun showSettingsDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_settings, null)
        val apiInput = dialogView.findViewById<androidx.appcompat.widget.AppCompatEditText>(R.id.et_api_url)
        val modelInput = dialogView.findViewById<androidx.appcompat.widget.AppCompatEditText>(R.id.et_model)

        // Load current config
        apiInput.setText(Config.apiUrl)
        modelInput.setText(Config.model)

        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("Settings")
            .setView(dialogView)
            .setPositiveButton("Save") { _, _ ->
                val newApiUrl = apiInput.text.toString().trim()
                val newModel = modelInput.text.toString().trim()
                if (newApiUrl.isNotEmpty() && newModel.isNotEmpty()) {
                    Config.apiUrl = newApiUrl
                    Config.model = newModel
                    Config.save(filesDir.absolutePath)
                    updateStatus("Settings saved - API: $newApiUrl")
                    Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Please fill all fields", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .create()

        dialog.show()
    }

    private fun setupLocalServerControls(
        btnStartServer: android.widget.Button,
        btnStopServer: android.widget.Button,
        serverStatusText: android.widget.TextView
    ) {
        // Local Settings button - opens settings dialog
        findViewById<android.widget.Button>(R.id.btn_server_settings).setOnClickListener {
            showServerSettingsDialog()
        }

        // Start Local button - loads config from SharedPreferences and initializes local inference backend
        btnStartServer.setOnClickListener {
            // Load config from SharedPreferences
            val modelPath = selectedModelPath
            val mmprojPath = selectedMmprojPath
            if (modelPath == null || mmprojPath == null) {
                Toast.makeText(this, "Please configure and save settings first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val serverConfig = loadServerConfig()
            val backend = currentBackend

            i( "=== Start Local button clicked ===")
            i( "  selectedModelPath: $modelPath")
            i( "  selectedMmprojPath: $mmprojPath")
            i( "  currentBackend: $backend")
            i( "  serverConfig.nCtx: ${serverConfig.nCtx}")
            i( "  serverConfig.nBatch: ${serverConfig.nBatch}")
            i( "  serverConfig.verbose: ${serverConfig.verbose}")
            i( "  serverConfig.htpSessions: ${serverConfig.htpSessions}")

            // Destroy existing backend if any
            VlmApplication.localInferenceBackend?.destroy()

            // Initialize local inference backend on background thread to avoid UI freeze
            CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
                LocalInferenceService.setApplicationContext(this@MainActivity)
                val localBackend = LocalInferenceBackend(this@MainActivity)
                val success = localBackend.initialize(
                    config = InferenceBackend.BackendConfig(
                        modelPath = modelPath,
                        mmprojPath = mmprojPath,
                        backend = backend,
                        maxTokens = Config.maxTokens,
                        temperature = serverConfig.temperature,
                        topP = serverConfig.topP
                    ),
                    nCtx = serverConfig.nCtx,
                    nBatch = serverConfig.nBatch,
                    verbose = serverConfig.verbose,
                    htpSessions = serverConfig.htpSessions,
                    imageMinTokens = serverConfig.imageMinTokens,
                    topK = serverConfig.topK,
                    topP = serverConfig.topP,
                    temperature = serverConfig.temperature,
                    repetitionPenalty = serverConfig.repetitionPenalty
                )

                withContext(Dispatchers.Main) {
                    if (success) {
                        // Store in singleton for ScreenCaptureService to use
                        VlmApplication.localInferenceBackend = localBackend
                        val modelName = File(modelPath).name
                        val mmprojName = File(mmprojPath).name
                        serverStatusText.text = "Backend: ${backend.name} | Model: ${modelName} | MMProj: ${mmprojName}\nctx=${serverConfig.nCtx}, batch=${serverConfig.nBatch}, maxTokens=${Config.maxTokens}\ntopK=${serverConfig.topK}, topP=${serverConfig.topP}, temp=${serverConfig.temperature}, repPen=${serverConfig.repetitionPenalty}\nTap 'Start Analyzer' to begin capture"
                        serverStatusText.setTextColor(Color.parseColor("#4CAF50"))
                        updateStatus("Local inference loaded and ready - Tap 'Start Analyzer' to begin")
                    } else {
                        serverStatusText.text = "Local Inference: Failed to load model"
                        serverStatusText.setTextColor(Color.parseColor("#f44336"))
                        updateStatus("Failed to initialize local inference")
                    }
                }
            }
        }

        // Stop Local button - destroys local inference backend
        btnStopServer.setOnClickListener {
            VlmApplication.localInferenceBackend?.destroy()
            VlmApplication.localInferenceBackend = null
            serverStatusText.text = "Server: Stopped"
            serverStatusText.setTextColor(Color.parseColor("#f44336"))
            updateStatus("Local inference stopped")
        }
    }

    private fun saveLocalBackendConfig(modelPath: String, mmprojPath: String, backend: Backend) {
        val prefs = getSharedPreferences()
        val modeIndex = if (currentMode == BackendMode.LOCAL) 1 else 0
        prefs.edit()
            .putString(KEY_MODEL_PATH, modelPath)
            .putString(KEY_MMPROJ_PATH, mmprojPath)
            .putString(KEY_LOCAL_BACKEND, backend.name)
            .putInt(KEY_MODE, modeIndex)
            .apply()
        i( "=== saveLocalBackendConfig ===")
        i( "  modelPath: $modelPath")
        i( "  mmprojPath: $mmprojPath")
        i( "  backend: $backend")
        i( "  modeIndex: $modeIndex (saved together with backend config)")
    }

    data class ServerConfig(
        val port: Int = 8080,
        val nCtx: Int = 16384,
        val nBatch: Int = 512,
        val verbose: Boolean = false,
        val debugLogsEnabled: Boolean = false,
        val infoLogsEnabled: Boolean = false,
        val htpSessions: Int = 1,
        // Inference parameters
        val imageMinTokens: Int = -1,
        val topK: Int = 40,
        val topP: Float = 0.95f,
        val temperature: Float = 0.80f,
        val repetitionPenalty: Float = 1.00f
    )

    private fun loadServerConfig(): ServerConfig {
        val prefs = getSharedPreferences()
        val nCtx = prefs.getInt(KEY_SERVER_N_CTX, 16384)
        val nBatch = prefs.getInt(KEY_SERVER_N_BATCH, 512)
        val verbose = prefs.getBoolean(KEY_SERVER_VERBOSE, false)
        val debugLogsEnabled = prefs.getBoolean(KEY_DEBUG_LOGS_ENABLED, false)
        val infoLogsEnabled = prefs.getBoolean(KEY_INFO_LOGS_ENABLED, false)
        val htpSessions = prefs.getInt(KEY_HTP_SESSIONS, 1)
        val imageMinTokens = prefs.getInt(KEY_IMAGE_MIN_TOKENS, -1)
        val topK = prefs.getInt(KEY_TOP_K, 40)
        val topP = prefs.getFloat(KEY_TOP_P, 0.95f)
        val temperature = prefs.getFloat(KEY_TEMPERATURE, 0.80f)
        val repetitionPenalty = prefs.getFloat(KEY_REPETITION_PENALTY, 1.00f)

        i( "=== loadServerConfig ===")
        i( "  KEY_SERVER_N_CTX: $nCtx")
        i( "  KEY_SERVER_N_BATCH: $nBatch")
        i( "  KEY_SERVER_VERBOSE: $verbose")
        i( "  KEY_DEBUG_LOGS_ENABLED: $debugLogsEnabled")
        i( "  KEY_HTP_SESSIONS: $htpSessions")
        i( "  KEY_IMAGE_MIN_TOKENS: $imageMinTokens")
        i( "  KEY_TOP_K: $topK")
        i( "  KEY_TOP_P: $topP")
        i( "  KEY_TEMPERATURE: $temperature")
        i( "  KEY_REPETITION_PENALTY: $repetitionPenalty")

        return ServerConfig(
            port = 8080,
            nCtx = nCtx,
            nBatch = nBatch,
            verbose = verbose,
            debugLogsEnabled = debugLogsEnabled,
            infoLogsEnabled = infoLogsEnabled,
            htpSessions = htpSessions,
            imageMinTokens = imageMinTokens,
            topK = topK,
            topP = topP,
            temperature = temperature,
            repetitionPenalty = repetitionPenalty
        )
    }

    private fun saveServerConfig(config: ServerConfig) {
        val prefs = getSharedPreferences()
        i( "=== saveServerConfig ===")
        i( "  nCtx: ${config.nCtx}")
        i( "  nBatch: ${config.nBatch}")
        i( "  verbose: ${config.verbose}")
        i( "  debugLogsEnabled: ${config.debugLogsEnabled}")
        i( "  htpSessions: ${config.htpSessions}")
        i( "  imageMinTokens: ${config.imageMinTokens}")
        i( "  topK: ${config.topK}")
        i( "  topP: ${config.topP}")
        i( "  temperature: ${config.temperature}")
        i( "  repetitionPenalty: ${config.repetitionPenalty}")
        prefs.edit()
            .putInt(KEY_SERVER_N_CTX, config.nCtx)
            .putInt(KEY_SERVER_N_BATCH, config.nBatch)
            .putBoolean(KEY_SERVER_VERBOSE, config.verbose)
            .putBoolean(KEY_DEBUG_LOGS_ENABLED, config.debugLogsEnabled)
            .putBoolean(KEY_INFO_LOGS_ENABLED, config.infoLogsEnabled)
            .putInt(KEY_HTP_SESSIONS, config.htpSessions)
            .putInt(KEY_IMAGE_MIN_TOKENS, config.imageMinTokens)
            .putInt(KEY_TOP_K, config.topK)
            .putFloat(KEY_TOP_P, config.topP)
            .putFloat(KEY_TEMPERATURE, config.temperature)
            .putFloat(KEY_REPETITION_PENALTY, config.repetitionPenalty)
            .apply()
    }

    private fun showServerSettingsDialog() {
        var config = loadServerConfig()
        val dialogView = layoutInflater.inflate(R.layout.dialog_server_settings, null)

        // Backend buttons
        val btnBackendCpu = dialogView.findViewById<android.widget.Button>(R.id.btn_backend_cpu)
        val btnBackendGpu = dialogView.findViewById<android.widget.Button>(R.id.btn_backend_gpu)
        val btnBackendNpu = dialogView.findViewById<android.widget.Button>(R.id.btn_backend_npu)

        // Model/MMProj buttons
        val btnSelectModel = dialogView.findViewById<android.widget.Button>(R.id.btn_select_model)
        val btnSelectMmproj = dialogView.findViewById<android.widget.Button>(R.id.btn_select_mmproj)

        // Settings inputs
        val ctxInput = dialogView.findViewById<android.widget.EditText>(R.id.et_server_n_ctx)
        val batchInput = dialogView.findViewById<android.widget.EditText>(R.id.et_server_n_batch)
        val verboseCheck = dialogView.findViewById<android.widget.CheckBox>(R.id.cb_server_verbose)
        val maxTokensInput = dialogView.findViewById<android.widget.EditText>(R.id.et_server_max_tokens)
        val htpSessionsSpinner = dialogView.findViewById<android.widget.Spinner>(R.id.spinner_htp_sessions)
        // Inference parameters
        val imageMinTokensInput = dialogView.findViewById<android.widget.EditText>(R.id.et_image_min_tokens)
        val topKInput = dialogView.findViewById<android.widget.EditText>(R.id.et_top_k)
        val topPInput = dialogView.findViewById<android.widget.EditText>(R.id.et_top_p)
        val temperatureInput = dialogView.findViewById<android.widget.EditText>(R.id.et_temperature)
        val repetitionPenaltyInput = dialogView.findViewById<android.widget.EditText>(R.id.et_repetition_penalty)

        // Pre-fill with current values
        ctxInput.setText(config.nCtx.toString())
        batchInput.setText(config.nBatch.toString())
        verboseCheck.isChecked = config.verbose
        maxTokensInput.setText(Config.maxTokens.toString())
        imageMinTokensInput.setText(if (config.imageMinTokens < 0) "" else config.imageMinTokens.toString())
        topKInput.setText(config.topK.toString())
        topPInput.setText(String.format("%.2f", config.topP))
        temperatureInput.setText(String.format("%.2f", config.temperature))
        repetitionPenaltyInput.setText(String.format("%.2f", config.repetitionPenalty))

        // Setup HTP sessions spinner
        val htpSessionsAdapter = android.widget.ArrayAdapter(this, android.R.layout.simple_spinner_item, arrayOf("1", "2", "3", "4"))
        htpSessionsAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        htpSessionsSpinner.adapter = htpSessionsAdapter
        htpSessionsSpinner.setSelection((config.htpSessions - 1).coerceAtLeast(0))

        // Debug logs toggle checkbox
        val cbDebugLogs = dialogView.findViewById<android.widget.CheckBox>(R.id.cb_debug_logs)
        // Set initial state without triggering listener
        cbDebugLogs.setOnCheckedChangeListener(null)
        cbDebugLogs.isChecked = config.debugLogsEnabled
        // Now add the listener for user changes only
        cbDebugLogs.setOnCheckedChangeListener { _, isChecked ->
            i( "=== DEBUG CHECKBOX user changed to: $isChecked ===")
            VlmApplication.debugLogsEnabled = isChecked
            nativeSetDebugLogsEnabled(isChecked)
            i( "=== After setting: VlmApplication.debugLogsEnabled = ${VlmApplication.debugLogsEnabled} ===")
        }

        // Info logs toggle checkbox
        val cbInfoLogs = dialogView.findViewById<android.widget.CheckBox>(R.id.cb_info_logs)
        cbInfoLogs.setOnCheckedChangeListener(null)
        cbInfoLogs.isChecked = config.infoLogsEnabled
        cbInfoLogs.setOnCheckedChangeListener { _, isChecked ->
            i( "=== INFO CHECKBOX user changed to: $isChecked ===")
            VlmApplication.infoLogsEnabled = isChecked
            nativeSetInfoLogsEnabled(isChecked)
            i( "=== After setting: VlmApplication.infoLogsEnabled = ${VlmApplication.infoLogsEnabled} ===")
        }

        // Backend selection buttons
        btnBackendCpu.setOnClickListener {
            currentBackend = Backend.CPU
            updateDialogBackendButtons(dialogView)
        }
        btnBackendGpu.setOnClickListener {
            currentBackend = Backend.GPU
            updateDialogBackendButtons(dialogView)
        }
        btnBackendNpu.setOnClickListener {
            currentBackend = Backend.NPU
            updateDialogBackendButtons(dialogView)
        }
        updateDialogBackendButtons(dialogView)

        // Model picker
        btnSelectModel.setOnClickListener {
            openFilePicker(isModel = true, dialogView = dialogView)
        }

        // MMProj picker
        btnSelectMmproj.setOnClickListener {
            openFilePicker(isModel = false, dialogView = dialogView)
        }

        // Update display with current values
        val tvSelectedModel = dialogView.findViewById<TextView>(R.id.tv_selected_model)
        val tvSelectedMmproj = dialogView.findViewById<TextView>(R.id.tv_selected_mmproj)
        tvSelectedModel.text = selectedModelPath?.let { File(it).name } ?: "None"
        tvSelectedModel.setTextColor(Color.parseColor(if (selectedModelPath != null) "#ffffff" else "#888888"))
        tvSelectedMmproj.text = selectedMmprojPath?.let { File(it).name } ?: "None"
        tvSelectedMmproj.setTextColor(Color.parseColor(if (selectedMmprojPath != null) "#ffffff" else "#888888"))

        val dialog = AlertDialog.Builder(this)
            .setTitle("Local Settings")
            .setView(dialogView)
            .setPositiveButton("Save") { _, _ ->
                val newCtx = ctxInput.text.toString().toIntOrNull() ?: 16384
                val newBatch = batchInput.text.toString().toIntOrNull() ?: 512
                val newVerbose = verboseCheck.isChecked
                val newDebugLogsEnabled = cbDebugLogs.isChecked
                val newInfoLogsEnabled = cbInfoLogs.isChecked
                val newMaxTokens = maxTokensInput.text.toString().toIntOrNull() ?: 1024
                val newHtpSessions = htpSessionsSpinner.selectedItemPosition + 1
                val newImageMinTokens = imageMinTokensInput.text.toString().toIntOrNull() ?: -1
                val newTopK = topKInput.text.toString().toIntOrNull() ?: 40
                val newTopP = topPInput.text.toString().toFloatOrNull() ?: 0.95f
                val newTemperature = temperatureInput.text.toString().toFloatOrNull() ?: 0.80f
                val newRepetitionPenalty = repetitionPenaltyInput.text.toString().toFloatOrNull() ?: 1.00f

                // Save all settings
                if (selectedModelPath != null && selectedMmprojPath != null) {
                    saveLocalBackendConfig(selectedModelPath!!, selectedMmprojPath!!, currentBackend)
                    // Update VlmApplication for camera mode access
                    VlmApplication.modelPath = selectedModelPath
                    VlmApplication.mmprojPath = selectedMmprojPath
                    VlmApplication.backendType = currentBackend
                }
                val serverConfig = ServerConfig(
                    port = 8080,
                    nCtx = newCtx,
                    nBatch = newBatch,
                    verbose = newVerbose,
                    debugLogsEnabled = newDebugLogsEnabled,
                    infoLogsEnabled = newInfoLogsEnabled,
                    htpSessions = newHtpSessions,
                    imageMinTokens = newImageMinTokens,
                    topK = newTopK,
                    topP = newTopP,
                    temperature = newTemperature,
                    repetitionPenalty = newRepetitionPenalty
                )
                saveServerConfig(serverConfig)
                Config.maxTokens = newMaxTokens
                Config.save(filesDir.absolutePath)

                // Update native debug logs flag
                VlmApplication.debugLogsEnabled = newDebugLogsEnabled
                nativeSetDebugLogsEnabled(newDebugLogsEnabled)
                VlmApplication.infoLogsEnabled = newInfoLogsEnabled
                nativeSetInfoLogsEnabled(newInfoLogsEnabled)

                updateStatus("Local settings saved: ctx=$newCtx, batch=$newBatch, topK=$newTopK, topP=$newTopP, temp=$newTemperature, repPen=$newRepetitionPenalty")
                Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .create()

        dialog.show()
    }

    private fun updateDialogBackendButtons(dialogView: android.view.View) {
        val btnBackendCpu = dialogView.findViewById<android.widget.Button>(R.id.btn_backend_cpu)
        val btnBackendGpu = dialogView.findViewById<android.widget.Button>(R.id.btn_backend_gpu)
        val btnBackendNpu = dialogView.findViewById<android.widget.Button>(R.id.btn_backend_npu)

        val defaultColor = Color.parseColor("#666666")
        val activeColor = Color.parseColor("#4CAF50")

        btnBackendCpu.backgroundTintList = android.content.res.ColorStateList.valueOf(if (currentBackend == Backend.CPU) activeColor else defaultColor)
        btnBackendGpu.backgroundTintList = android.content.res.ColorStateList.valueOf(if (currentBackend == Backend.GPU) activeColor else defaultColor)
        btnBackendNpu.backgroundTintList = android.content.res.ColorStateList.valueOf(if (currentBackend == Backend.NPU) activeColor else defaultColor)
    }

    private var dialogFilePickerCallback: ((String) -> Unit)? = null

    private fun loadLocalBackendConfig() {
        val prefs = getSharedPreferences()
        val modelPath = prefs.getString(KEY_MODEL_PATH, null)
        val mmprojPath = prefs.getString(KEY_MMPROJ_PATH, null)
        val backendName = prefs.getString(KEY_LOCAL_BACKEND, Backend.CPU.name)
        val modeIndex = prefs.getInt(KEY_MODE, 0)

        i( "=== loadLocalBackendConfig ===")
        i( "  KEY_MODEL_PATH: $modelPath")
        i( "  KEY_MMPROJ_PATH: $mmprojPath")
        i( "  KEY_LOCAL_BACKEND: $backendName")
        i( "  KEY_MODE: $modeIndex (0=REMOTE, 1=LOCAL)")

        // Load saved mode
        currentMode = if (modeIndex == 0) BackendMode.REMOTE else BackendMode.LOCAL
        i( "  currentMode set to: $currentMode")

        // Set selected paths directly (full absolute paths saved)
        if (modelPath != null) {
            selectedModelPath = modelPath
        }
        if (mmprojPath != null) {
            selectedMmprojPath = mmprojPath
        }

        // Load saved backend
        currentBackend = try {
            Backend.valueOf(backendName ?: Backend.CPU.name)
        } catch (e: Exception) {
            Backend.CPU
        }

        // Update VlmApplication for camera mode access
        VlmApplication.modelPath = selectedModelPath
        VlmApplication.mmprojPath = selectedMmprojPath
        VlmApplication.backendType = currentBackend
    }


    private var pickIntentIsModel: Boolean = true
    private var currentBrowseDir: String = MODELS_DIR

    private fun openFilePicker(isModel: Boolean, dialogView: android.view.View? = null) {
        pickIntentIsModel = isModel
        currentBrowseDir = MODELS_DIR
        i( "Opening file picker for ${if (isModel) "Model" else "MMProj"}, starting at: $currentBrowseDir")
        if (dialogView != null) {
            showFileBrowserForDialog(dialogView)
        } else {
            showFileBrowser()
        }
    }

    private fun showFileBrowserForDialog(dialogView: android.view.View) {
        val dir = File(currentBrowseDir)
        i( "showFileBrowserForDialog: checking dir=${currentBrowseDir}, exists=${dir.exists()}")

        if (!dir.exists()) {
            Log.e(TAG, "Directory does not exist: $currentBrowseDir")
            Toast.makeText(this, "Directory not found: $currentBrowseDir\nPlease push models via adb", Toast.LENGTH_LONG).show()
            return
        }

        if (!dir.isDirectory) {
            Log.e(TAG, "Path is not a directory: $currentBrowseDir")
            Toast.makeText(this, "Not a directory", Toast.LENGTH_SHORT).show()
            return
        }

        val filesArray = dir.listFiles()
        i( "listFiles returned: ${filesArray?.size ?: -1} items, array=null?${filesArray == null}")
        if (filesArray == null) {
            Log.e(TAG, "Cannot read directory: $currentBrowseDir")
            Toast.makeText(this, "Cannot read directory: $currentBrowseDir\nCheck storage permissions", Toast.LENGTH_SHORT).show()
            return
        }

        val files = filesArray.asList().sortedBy { !it.isFile || !it.name.endsWith(".gguf", ignoreCase = true) }

        val displayNames = mutableListOf<String>()
        val fileEntries = mutableListOf<File>()

        // Add parent directory option if not at root
        if (dir.parent != null) {
            displayNames.add(".. (go up)")
            fileEntries.add(File(dir.parent!!))
        }

        // Add directories first
        for (f in files) {
            if (f.isDirectory) {
                displayNames.add("[DIR] ${f.name}")
                fileEntries.add(f)
            }
        }

        // Add .gguf files
        for (f in files) {
            if (f.isFile && f.name.endsWith(".gguf", ignoreCase = true)) {
                displayNames.add("[FILE] ${f.name}")
                fileEntries.add(f)
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Select ${if (pickIntentIsModel) "Model" else "MMProj"}")
            .setItems(displayNames.toTypedArray()) { _, which ->
                val selected = fileEntries[which]
                if (selected.isDirectory) {
                    if (selected.name == "..") {
                        currentBrowseDir = selected.parent ?: MODELS_DIR
                    } else {
                        currentBrowseDir = selected.absolutePath
                    }
                    showFileBrowserForDialog(dialogView)
                } else {
                    // Copy selected file to app's private storage (deletes old model/mmproj first)
                    val privatePath = copyFileToPrivateStorage(selected.absolutePath, pickIntentIsModel)

                    if (pickIntentIsModel) {
                        selectedModelPath = privatePath
                        VlmApplication.modelPath = privatePath
                    } else {
                        selectedMmprojPath = privatePath
                        VlmApplication.mmprojPath = privatePath
                    }

                    // Update dialog display
                    val tvSelectedModel = dialogView.findViewById<TextView>(R.id.tv_selected_model)
                    val tvSelectedMmproj = dialogView.findViewById<TextView>(R.id.tv_selected_mmproj)
                    tvSelectedModel.text = selectedModelPath?.let { File(it).name } ?: "None"
                    tvSelectedModel.setTextColor(Color.parseColor(if (selectedModelPath != null) "#ffffff" else "#888888"))
                    tvSelectedMmproj.text = selectedMmprojPath?.let { File(it).name } ?: "None"
                    tvSelectedMmproj.setTextColor(Color.parseColor(if (selectedMmprojPath != null) "#ffffff" else "#888888"))

                    Toast.makeText(this, "Selected: ${selected.name}", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showFileBrowser() {
        val dir = File(currentBrowseDir)
        i( "showFileBrowser: checking dir=${currentBrowseDir}, exists=${dir.exists()}")

        if (!dir.exists()) {
            Log.e(TAG, "Directory does not exist: $currentBrowseDir")
            Toast.makeText(this, "Directory not found: $currentBrowseDir\nPlease push models via adb", Toast.LENGTH_LONG).show()
            return
        }

        if (!dir.isDirectory) {
            Log.e(TAG, "Path is not a directory: $currentBrowseDir")
            Toast.makeText(this, "Not a directory", Toast.LENGTH_SHORT).show()
            return
        }

        val filesArray = dir.listFiles()
        i( "listFiles returned: ${filesArray?.size ?: -1} items, array=null?${filesArray == null}")
        if (filesArray == null) {
            Log.e(TAG, "Cannot read directory: $currentBrowseDir")
            Toast.makeText(this, "Cannot read directory: $currentBrowseDir\nCheck storage permissions", Toast.LENGTH_SHORT).show()
            return
        }

        val files = filesArray.asList().sortedBy { !it.isFile || !it.name.endsWith(".gguf", ignoreCase = true) }

        val displayNames = mutableListOf<String>()
        val fileEntries = mutableListOf<File>()

        // Add parent directory option if not at root
        if (dir.parent != null) {
            displayNames.add(".. (go up)")
            fileEntries.add(File(dir.parent!!))
        }

        // Add directories first
        for (f in files) {
            if (f.isDirectory) {
                displayNames.add("[DIR] ${f.name}")
                fileEntries.add(f)
            }
        }

        // Add .gguf files with filtering based on picker type
        for (f in files) {
            if (f.isFile && f.name.endsWith(".gguf", ignoreCase = true)) {
                val isMmproj = f.name.contains("mmproj", ignoreCase = true)
                // Filter: Model picker shows models, MMProj picker shows mmprojs
                val shouldShow = if (pickIntentIsModel) {
                    !isMmproj  // Model picker: show files that are NOT mmproj
                } else {
                    isMmproj   // MMProj picker: show files that ARE mmproj
                }
                if (shouldShow) {
                    displayNames.add("[FILE] ${f.name}")
                    fileEntries.add(f)
                }
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Select ${if (pickIntentIsModel) "Model" else "MMProj"}")
            .setItems(displayNames.toTypedArray()) { _, which ->
                val selected = fileEntries[which]
                if (selected.isDirectory) {
                    if (selected.name == "..") {
                        currentBrowseDir = selected.parent ?: MODELS_DIR
                    } else {
                        currentBrowseDir = selected.absolutePath
                    }
                    showFileBrowser()
                } else {
                    // Copy selected file to app's private storage (deletes old model/mmproj first)
                    val privatePath = copyFileToPrivateStorage(selected.absolutePath, pickIntentIsModel)

                    if (pickIntentIsModel) {
                        selectedModelPath = privatePath
                        VlmApplication.modelPath = privatePath
                    } else {
                        selectedMmprojPath = privatePath
                        VlmApplication.mmprojPath = privatePath
                    }
                    VlmApplication.backendType = currentBackend
                    Toast.makeText(this, "Selected: ${selected.name}", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun saveMode() {
        val prefs = getSharedPreferences()
        val modeIndex = if (currentMode == BackendMode.LOCAL) 1 else 0
        i( "=== saveMode ===")
        i( "  currentMode: $currentMode")
        i( "  modeIndex: $modeIndex")
        prefs.edit().putInt(KEY_MODE, modeIndex).apply()
        i( "  Mode saved to SharedPreferences")
    }

    private fun updateModeButtons() {
        val btnModeRemote = findViewById<android.widget.Button>(R.id.btn_mode_remote)
        val btnModeLocal = findViewById<android.widget.Button>(R.id.btn_mode_local)

        i( "=== updateModeButtons ===")
        i( "  currentMode: $currentMode")

        if (currentMode == BackendMode.REMOTE) {
            btnModeRemote.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#4CAF50"))
            btnModeLocal.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#666666"))
        } else {
            btnModeRemote.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#666666"))
            btnModeLocal.backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#4CAF50"))
        }
    }

    private var currentBackend: Backend = Backend.CPU

    private fun updateSettingsButtonVisibility() {
        val btnSettings = findViewById<android.widget.Button>(R.id.btn_settings)
        val btnServerSettings = findViewById<android.widget.Button>(R.id.btn_server_settings)
        i( "=== updateSettingsButtonVisibility ===")
        i( "  currentMode: $currentMode")
        if (currentMode == BackendMode.REMOTE) {
            btnSettings.visibility = android.view.View.VISIBLE
            btnServerSettings.visibility = android.view.View.GONE
        } else {
            btnSettings.visibility = android.view.View.GONE
            btnServerSettings.visibility = android.view.View.VISIBLE
        }
    }

    private fun updateBackendButtons() {
        val btnBackendCpu = findViewById<android.widget.Button>(R.id.btn_backend_cpu)
        val btnBackendGpu = findViewById<android.widget.Button>(R.id.btn_backend_gpu)
        val btnBackendNpu = findViewById<android.widget.Button>(R.id.btn_backend_npu)

        i( "=== updateBackendButtons ===")
        i( "  currentBackend: $currentBackend")

        val defaultColor = Color.parseColor("#666666")
        val activeColor = Color.parseColor("#4CAF50")

        btnBackendCpu.backgroundTintList = android.content.res.ColorStateList.valueOf(if (currentBackend == Backend.CPU) activeColor else defaultColor)
        btnBackendGpu.backgroundTintList = android.content.res.ColorStateList.valueOf(if (currentBackend == Backend.GPU) activeColor else defaultColor)
        btnBackendNpu.backgroundTintList = android.content.res.ColorStateList.valueOf(if (currentBackend == Backend.NPU) activeColor else defaultColor)
    }

    override fun onDestroy() {
        super.onDestroy()
        i( "MainActivity destroyed")
        if (isServiceRunning) {
            stopCapture()
        }
        if (isOverlayEnabled) {
            OverlayViewService.stopService(this)
            isOverlayEnabled = false
        }
        // Stop local inference backends
        localBackend?.destroy()
        // Don't destroy VlmApplication.localInferenceBackend here - it's managed by "Stop Local" button
        unregisterBroadcastReceiver()
        ttsHelper.shutdown()
        nativeDestroy()
    }

    private fun registerBroadcastReceiver() {
        broadcastReceiver = object : android.content.BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                when (intent.action) {
                    ScreenCaptureService.ACTION_ANALYSIS_STREAM_COMPLETE -> {
                        val result = intent.getStringExtra(ScreenCaptureService.EXTRA_STREAM_CONTENT)
                        val promptTokens = intent.getIntExtra(ScreenCaptureService.EXTRA_RESULT_PROMPT_TOKENS, 0)
                        val completionTokens = intent.getIntExtra(ScreenCaptureService.EXTRA_RESULT_COMPLETION_TOKENS, 0)
                        val totalTimeMs = intent.getLongExtra(ScreenCaptureService.EXTRA_RESULT_TOTAL_TIME_MS, 0L)
                        val imagePath = intent.getStringExtra(ScreenCaptureService.EXTRA_IMAGE_PATH)
                        val tokensPerSecond = intent.getDoubleExtra(ScreenCaptureService.EXTRA_RESULT_TOKENS_PER_SECOND, 0.0)
                        result?.let {
                            onAnalysisResult(it, promptTokens, completionTokens, totalTimeMs, tokensPerSecond, imagePath)
                        }
                    }
                    ScreenCaptureService.ACTION_ANALYSIS_ERROR -> {
                        val error = intent.getStringExtra(ScreenCaptureService.EXTRA_ERROR_MESSAGE)
                        error?.let {
                            onAnalysisError(it)
                        }
                    }
                    CameraCaptureActivity.ACTION_CAMERA_RESOLUTION_READY -> {
                        // Received actual camera resolution from CameraCaptureActivity
                        cameraWidth = intent.getIntExtra(CameraCaptureActivity.EXTRA_WIDTH, 0)
                        cameraHeight = intent.getIntExtra(CameraCaptureActivity.EXTRA_HEIGHT, 0)
                        if (cameraWidth > 0 && cameraHeight > 0) {
                            i("Received camera resolution: ${cameraWidth}x${cameraHeight}")
                            saveCameraResolution(cameraWidth, cameraHeight)
                            // Update preview if camera mode is selected
                            if (captureSource == CaptureSource.CAMERA) {
                                updateImageSizePreview(Config.imageScaleFactorCamera)
                            }
                        }
                    }
                }
            }
        }

        val filter = IntentFilter().apply {
            addAction(ScreenCaptureService.ACTION_ANALYSIS_STREAM_COMPLETE)
            addAction(ScreenCaptureService.ACTION_ANALYSIS_ERROR)
            addAction(CameraCaptureActivity.ACTION_CAMERA_RESOLUTION_READY)
        }
        val receiver = broadcastReceiver
        if (receiver != null) {
            LocalBroadcastManager.getInstance(this).registerReceiver(receiver, filter)
            i( "BroadcastReceiver registered")
        }
    }

    private fun unregisterBroadcastReceiver() {
        val receiver = broadcastReceiver
        if (receiver != null) {
            LocalBroadcastManager.getInstance(this).unregisterReceiver(receiver)
            broadcastReceiver = null
            i( "BroadcastReceiver unregistered")
        }
    }

    external fun nativeInit(assets: android.content.res.AssetManager, storagePath: String)
    external fun nativeProcessFrame(buffer: ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int)
    external fun nativeDestroy()

    // JNI function to set debug logs enabled flag (called from Kotlin when button is clicked)
    external fun nativeSetDebugLogsEnabled(enabled: Boolean)
    external fun nativeSetInfoLogsEnabled(enabled: Boolean)

    /**
     * Update the image size preview text based on current capture source and scale.
     * Called when capture source changes or camera resolution is discovered.
     */
    private fun updateImageSizePreview(scale: Float) {
        if (!::imageSizePreviewText.isInitialized) return  // UI not initialized yet

        val (baseWidth, baseHeight) = if (captureSource == CaptureSource.CAMERA) {
            // Use camera dimensions if available, otherwise show target resolution
            if (cameraWidth > 0 && cameraHeight > 0) {
                Pair(cameraWidth, cameraHeight)
            } else {
                // Default camera target resolution (1440×1080, 4:3 aspect ratio)
                Pair(1440, 1080)
            }
        } else {
            // Use screen dimensions for screen capture
            Pair(screenWidth, screenHeight)
        }
        val scaledWidth = (baseWidth * scale).toInt()
        val scaledHeight = (baseHeight * scale).toInt()
        imageSizePreviewText.text = "~${scaledWidth}x${scaledHeight}"
    }
}
