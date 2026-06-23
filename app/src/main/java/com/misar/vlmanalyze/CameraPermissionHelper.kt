// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * Helper for camera permission handling.
 * Requests camera permission once per app install (like audio permission).
 */
object CameraPermissionHelper {

    private const val PERMISSION_REQUEST_CODE = 201

    /**
     * Check if camera permission is granted.
     */
    fun hasCameraPermission(context: Context): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Request camera permission from the user.
     * Shows a toast if permission is denied.
     *
     * @return true if permission is already granted, false if request was shown
     */
    fun requestCameraPermission(activity: AppCompatActivity): Boolean {
        if (hasCameraPermission(activity)) {
            return true
        }

        // Request permission
        ActivityCompat.requestPermissions(
            activity,
            arrayOf(Manifest.permission.CAMERA),
            PERMISSION_REQUEST_CODE
        )

        return false
    }

    /**
     * Handle the permission request result.
     * Call this from MainActivity.onRequestPermissionsResult()
     *
     * @return true if permission was granted, false if denied
     */
    fun handlePermissionResult(
        context: Context,
        requestCode: Int,
        grantResults: IntArray,
        onError: () -> Unit = {}
    ): Boolean {
        if (requestCode != PERMISSION_REQUEST_CODE) {
            return false
        }

        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            // Permission granted
            return true
        } else {
            // Permission denied - show toast
            Toast.makeText(
                context,
                "Camera permission denied. Camera mode requires camera access.",
                Toast.LENGTH_LONG
            ).show()
            onError()
            return false
        }
    }
}
