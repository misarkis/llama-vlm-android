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
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.*
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground service that runs llama-server as a persistent HTTP server
 * for local VLM inference on device.
 *
 * Approach (Phase 4 - JNI Library):
 * - libllama-server-impl.so built by build-llamacpp-android.sh (41 MB)
 * - libllama-server-jni.so compiled by app CMake from server-jni.cpp (35 KB)
 * - All libraries copied to app/src/main/jniLibs/arm64-v8a/ by build script
 * - Android extracts .so files to nativeLibraryDir at install time
 * - System.loadLibrary() loads the libraries and nativeStartServer() starts HTTP server
 *
 * Why this works:
 * - Android's package manager extracts all .so files from jniLibs to nativeLibraryDir
 * - nativeLibraryDir is executable - Android loads .so files from there
 * - libllama-server-impl.so contains the full server logic (same as llama-server executable)
 * - libllama-server-jni.so provides JNI bridge to call server functions
 *
 * Why other approaches failed:
 * - filesDir: Android 10+ (API 29+) mounts /data/data/<package>/files/ as noexec
 * - jniLibs executables: Android only extracts ELF ET_DYN (shared libs), not ELF ET_EXEC (executables)
 * - /data/local/tmp/: Not writable by app (only shell/root)
 *
 * Environment variables per backend (set in server-jni.cpp before llama_server()):
 * - CPU: LD_LIBRARY_PATH=./lib:./bin:/vendor/lib64, --n-gpu-layers 0
 * - GPU: LD_LIBRARY_PATH=./lib:./bin:/vendor/lib64, --n-gpu-layers 999
 * - NPU: LD_LIBRARY_PATH=./lib:./bin:/vendor/lib64, GGML_HEXAGON_NDEV=4, --device HTP0,HTP1,HTP2,HTP3, --n-gpu-layers 999
 */
class LlamaServerService : Service() {

    companion object {
        init {
            // Load llama.cpp server JNI libraries
            System.loadLibrary("llama-server-impl")
            System.loadLibrary("llama-server-jni")
        }

        private const val TAG = "VLM-LlamaServerSvc"
        private const val NOTIFICATION_ID = 2
        private const val CHANNEL_ID = "vlm_llama_server_channel"
        private const val DEFAULT_SERVER_PORT = 8080
        private const val DEFAULT_N_CTX = 4096
        private const val DEFAULT_N_BATCH = 512
        private const val DEFAULT_N_GPU_LAYERS = 999

        // Info log helper - only logs when info mode is enabled
        private fun i(msg: String) {
            if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
        }

        @Volatile
        private var sIsRunning = false

        fun isServiceRunning(): Boolean = sIsRunning

        fun startService(
            context: Context,
            modelPath: String,
            mmprojPath: String,
            backend: Backend,
            port: Int = DEFAULT_SERVER_PORT,
            nCtx: Int = DEFAULT_N_CTX,
            nBatch: Int = DEFAULT_N_BATCH,
            nGpuLayers: Int = DEFAULT_N_GPU_LAYERS,
            verbose: Boolean = false,
            htpSessions: Int = 1
        ) {
            i( "startService called: model=$modelPath, backend=$backend, port=$port, htpSessions=$htpSessions")

            val intent = Intent(context, LlamaServerService::class.java).apply {
                putExtra(EXTRA_MODEL_PATH, modelPath)
                putExtra(EXTRA_MMPROJ_PATH, mmprojPath)
                putExtra(EXTRA_BACKEND, backend.name)
                putExtra(EXTRA_PORT, port)
                putExtra(EXTRA_N_CTX, nCtx)
                putExtra(EXTRA_N_BATCH, nBatch)
                putExtra(EXTRA_N_GPU_LAYERS, nGpuLayers)
                putExtra(EXTRA_VERBOSE, verbose)
                putExtra(EXTRA_HTP_SESSIONS, htpSessions)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stopService(context: Context) {
            i( "stopService called")
            context.stopService(Intent(context, LlamaServerService::class.java))
        }

        fun isServerReady(): Boolean = sIsRunning

        const val ACTION_START_SERVER = "com.misar.vlmanalyze.START_SERVER"

        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_MMPROJ_PATH = "mmproj_path"
        const val EXTRA_BACKEND = "backend"
        const val EXTRA_PORT = "port"
        const val EXTRA_N_CTX = "n_ctx"
        const val EXTRA_N_BATCH = "n_batch"
        const val EXTRA_N_GPU_LAYERS = "n_gpu_layers"
        const val EXTRA_VERBOSE = "verbose"
        const val EXTRA_HTP_SESSIONS = "htp_sessions"

        const val ACTION_SERVER_STARTED = "com.misar.vlmanalyze.SERVER_STARTED"
        const val ACTION_SERVER_STOPPED = "com.misar.vlmanalyze.SERVER_STOPPED"
        const val ACTION_SERVER_ERROR = "com.misar.vlmanalyze.SERVER_ERROR"
        const val EXTRA_ERROR_MESSAGE = "error_message"

        const val ACTION_STOP_SERVER = "com.misar.vlmanalyze.STOP_SERVER"
        const val ACTION_CHECK_STATUS = "com.misar.vlmanalyze.CHECK_STATUS"
    }

    // Native methods for JNI server (implemented in libllama-server-jni.so)
    external fun nativeStartServer(
        modelPath: String,
        mmprojPath: String,
        port: Int,
        nCtx: Int,
        nBatch: Int,
        nGpuLayers: Int,
        device: String?,
        verbose: Boolean
    ): Boolean

    external fun nativeStopServer()
    external fun nativeIsServerRunning(): Boolean

    private val handler = Handler(Looper.getMainLooper())
    private var isServerRunning = false
    private val isStopping = AtomicBoolean(false)
    private val tag = "VLM-LlamaServerSvc"
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // Debug log helper - only logs when debug mode is enabled
    private fun d(msg: String) {
        if (VlmApplication.debugLogsEnabled) Log.d(tag, msg)
    }

    // Info log helper - only logs when info mode is enabled
    private fun i(msg: String) {
        if (VlmApplication.infoLogsEnabled) Log.i(TAG, msg)
    }

    override fun onCreate() {
        super.onCreate()
        i( "LlamaServerService created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        i( "onStartCommand: ${intent?.action}")

        when (intent?.action) {
            ACTION_STOP_SERVER -> {
                stopServer()
                return START_NOT_STICKY
            }
            else -> {
                // Initial start - extract params and start server
                val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH)
                val mmprojPath = intent?.getStringExtra(EXTRA_MMPROJ_PATH)
                val backendName = intent?.getStringExtra(EXTRA_BACKEND) ?: "NPU"
                val port = intent?.getIntExtra(EXTRA_PORT, DEFAULT_SERVER_PORT) ?: DEFAULT_SERVER_PORT
                val nCtx = intent?.getIntExtra(EXTRA_N_CTX, DEFAULT_N_CTX) ?: DEFAULT_N_CTX
                val nBatch = intent?.getIntExtra(EXTRA_N_BATCH, DEFAULT_N_BATCH) ?: DEFAULT_N_BATCH
                val nGpuLayers = intent?.getIntExtra(EXTRA_N_GPU_LAYERS, DEFAULT_N_GPU_LAYERS) ?: DEFAULT_N_GPU_LAYERS
                val verbose = intent?.getBooleanExtra(EXTRA_VERBOSE, false) ?: false
                val htpSessions = intent?.getIntExtra(EXTRA_HTP_SESSIONS, 4) ?: 4

                if (modelPath == null || mmprojPath == null) {
                    Log.e(tag, "Missing model paths in intent")
                    startForegroundWithNotification("Error: Missing model paths")
                    stopSelf()
                    return START_NOT_STICKY
                }

                val backend = try {
                    Backend.valueOf(backendName)
                } catch (e: Exception) {
                    Log.e(tag, "Invalid backend: $backendName")
                    Backend.NPU
                }

                startForegroundWithNotification("Starting server...")
                startServer(modelPath, mmprojPath, backend, port, nCtx, nBatch, nGpuLayers, verbose, htpSessions)
            }
        }

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        i( "LlamaServerService destroying")
        serviceScope.cancel()
        nativeStopServer()
        isServerRunning = false
        sIsRunning = false
        handler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    private fun startForegroundWithNotification(statusText: String) {
        createNotificationChannel()

        val notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VLM Local Server")
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()

        try {
            startForeground(NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(tag, "Failed to start foreground", e)
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "VLM Local Server",
                NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    /**
     * Get the directory where models will be copied.
     * Uses context.filesDir/models/ for private storage.
     */
    private fun getModelDirectoryPath(): String {
        val filesDir = this.filesDir
        val modelsDir = File(filesDir, "models")
        modelsDir.mkdirs()
        return modelsDir.absolutePath
    }

    /**
     * Copy model file from source to app's private models directory.
     * Deletes only the old file of the same type before copying the new one.
     * Returns the new path in the app's private directory.
     */
    private fun copyModelToPrivateStorage(sourcePath: String, isModel: Boolean): String {
        val sourceFile = File(sourcePath)
        if (!sourceFile.exists()) {
            Log.e(tag, "Source file not found: $sourcePath")
            return sourcePath
        }

        val destDir = File(getModelDirectoryPath())
        destDir.mkdirs()

        // Delete only the old file of the same type (model deletes model, mmproj deletes mmproj)
        if (isModel) {
            val existingModels = destDir.listFiles()?.filter { it.name.endsWith("_model.gguf") || it.name.contains("Qwen") && it.extension == "gguf" } ?: emptyList()
            for (existingFile in existingModels) {
                existingFile.delete()
                i( "Deleted old model: ${existingFile.absolutePath}")
            }
        } else {
            val existingMmprojs = destDir.listFiles()?.filter { it.name.contains("mmproj") && it.extension == "gguf" } ?: emptyList()
            for (existingFile in existingMmprojs) {
                existingFile.delete()
                i( "Deleted old mmproj: ${existingFile.absolutePath}")
            }
        }

        val destFile = File(destDir, sourceFile.name)
        i( "Copying ${if (isModel) "model" else "mmproj"} from $sourcePath to ${destFile.absolutePath}")

        try {
            sourceFile.copyTo(destFile, overwrite = true)
            i( "Copied ${if (isModel) "model" else "mmproj"} successfully: ${destFile.absolutePath}")
            return destFile.absolutePath
        } catch (e: Exception) {
            Log.e(tag, "Failed to copy file: ${e.message}")
            return sourcePath
        }
    }

    private fun startServer(
        modelPath: String,
        mmprojPath: String,
        backend: Backend,
        port: Int = DEFAULT_SERVER_PORT,
        nCtx: Int = DEFAULT_N_CTX,
        nBatch: Int = DEFAULT_N_BATCH,
        nGpuLayers: Int = DEFAULT_N_GPU_LAYERS,
        verbose: Boolean = false,
        htpSessions: Int = 1
    ) {
        serviceScope.launch {
            try {
                // Copy model files to app's private storage (they may be in /data/local/tmp/ or /sdcard/ which need copying)
                i( "Checking if model files need to be copied to private storage...")
                val privateModelPath = if (modelPath.startsWith("/data/local/tmp/") || modelPath.startsWith("/sdcard/")) {
                    i( "Model in external storage, copying to private storage")
                    copyModelToPrivateStorage(modelPath, isModel = true)
                } else {
                    modelPath
                }

                val privateMmprojPath = if (mmprojPath.startsWith("/data/local/tmp/") || mmprojPath.startsWith("/sdcard/")) {
                    i( "MMProj in external storage, copying to private storage")
                    copyModelToPrivateStorage(mmprojPath, isModel = false)
                } else {
                    mmprojPath
                }

                // Verify model files exist (after copying if needed)
                if (!File(privateModelPath).exists()) {
                    Log.e(tag, "Model file not found: $privateModelPath")
                    withContext(Dispatchers.Main) {
                        notifyError("Model file not found: $privateModelPath")
                        stopSelf()
                    }
                    return@launch
                }
                if (!File(privateMmprojPath).exists()) {
                    Log.e(tag, "MMProj file not found: $privateMmprojPath")
                    withContext(Dispatchers.Main) {
                        notifyError("MMProj file not found: $privateMmprojPath")
                        stopSelf()
                    }
                    return@launch
                }

                // Backend configuration - override nGpuLayers if passed as parameter
                val actualGpuLayers = if (nGpuLayers != DEFAULT_N_GPU_LAYERS) {
                    nGpuLayers
                } else {
                    when (backend) {
                        Backend.CPU -> 0
                        Backend.GPU, Backend.NPU -> 999
                    }
                }

                val device = when (backend) {
                    Backend.NPU -> LocalInferenceService.buildHtpDeviceString(htpSessions)
                    else -> null
                }

                i( "Starting JNI server: model=$privateModelPath, mmproj=$privateMmprojPath, backend=$backend")
                i( "JNI params: port=$port, ctx=$nCtx, batch=$nBatch, ngl=$actualGpuLayers, device=$device, verbose=$verbose")

                // Start server via JNI (runs on native thread)
                val started = nativeStartServer(
                    modelPath = privateModelPath,
                    mmprojPath = privateMmprojPath,
                    port = port,
                    nCtx = nCtx,
                    nBatch = nBatch,
                    nGpuLayers = actualGpuLayers,
                    device = device,
                    verbose = verbose
                )

                if (started) {
                    isServerRunning = true
                    sIsRunning = true
                    i( "JNI server started successfully")

                    // Wait for server to be ready (on background thread)
                    delay(3000)
                    withContext(Dispatchers.Main) {
                        waitForServerReady(port)
                    }

                    withContext(Dispatchers.Main) {
                        notifyServerStarted()
                    }
                } else {
                    Log.e(tag, "Failed to start JNI server")
                    withContext(Dispatchers.Main) {
                        notifyError("Failed to start server via JNI")
                        stopSelf()
                    }
                }

            } catch (e: Exception) {
                Log.e(tag, "Failed to start server: ${e.javaClass.name} - ${e.message}")
                Log.e(tag, "Stack trace: ${e.stackTrace.joinToString("\n") { it.toString() }}")
                withContext(Dispatchers.Main) {
                    notifyError("Failed to start server: ${e.javaClass.simpleName} - ${e.message}")
                    stopSelf()
                }
            }
        }
    }

    private fun waitForServerReady(port: Int = DEFAULT_SERVER_PORT) {
        serviceScope.launch {
            val maxRetries = 45
            var retry = 0

            while (retry < maxRetries && isServerRunning && !isStopping.get()) {
                try {
                    val url = URL("http://127.0.0.1:$port/health")
                    val conn = url.openConnection() as HttpURLConnection
                    conn.connectTimeout = 1000
                    conn.readTimeout = 1000

                    val responseCode = conn.responseCode
                    conn.disconnect()

                    if (responseCode == 200) {
                        i( "Server is ready! Health check passed")
                        withContext(Dispatchers.Main) {
                            updateNotification("Server ready - Port $port")
                        }
                        return@launch
                    }
                } catch (e: Exception) {
                    // Server not ready yet
                }

                retry++
                d("Waiting for server... attempt $retry/$maxRetries")
                delay(1000)  // Wait 1 second before next attempt
            }

            if (retry >= maxRetries) {
                Log.e(tag, "Server failed to start within timeout")
                withContext(Dispatchers.Main) {
                    notifyError("Server failed to start within timeout")
                    stopSelf()
                }
            }
        }
    }

    private fun stopServer() {
        i( "Stopping JNI server")
        isStopping.set(true)

        // Stop server via JNI
        nativeStopServer()

        isServerRunning = false
        sIsRunning = false
        isStopping.set(false)

        notifyServerStopped()
        stopForeground(true)
        stopSelf()
    }

    private fun updateNotification(text: String) {
        val notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VLM Local Server")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()

        try {
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.notify(NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(tag, "Failed to update notification", e)
        }
    }

    private fun notifyServerStarted() {
        val intent = Intent(ACTION_SERVER_STARTED)
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun notifyServerStopped() {
        val intent = Intent(ACTION_SERVER_STOPPED)
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun notifyError(message: String) {
        Log.e(tag, "Server error: $message")
        val intent = Intent(ACTION_SERVER_ERROR).apply {
            putExtra(EXTRA_ERROR_MESSAGE, message)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }
}
