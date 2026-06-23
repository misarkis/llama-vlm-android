// Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
//
// This software is released under the MIT License.
// https://opensource.org

package com.misar.vlmanalyze

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream

/**
 * Utility functions for camera image processing.
 * Handles YUV to RGB conversion and image scaling.
 */
object CameraUtils {

    // Public log tags for logcat monitoring
    const val LOG_TAG_UTILS = "VLM-CameraUtils"
    const val LOG_TAG_CAMERA_CONVERT = "VLM-CameraConvert"

    // Debug log helper
    private fun d(msg: String) {
        if (VlmApplication.debugLogsEnabled) Log.d(LOG_TAG_UTILS, msg)
    }

    /**
     * Convert ImageProxy (YUV_420_888) to RGB byte array.
     * Uses CameraX's built-in toBitmap() extension for correct conversion.
     *
     * @param imageProxy The ImageProxy from CameraX (YUV_420_888 format)
     * @param rotationDegrees Rotation degrees (0, 90, 180, 270)
     * @return RGB byte array (width * height * 3 bytes)
     */
    fun imageProxyToRgbBytes(imageProxy: ImageProxy, rotationDegrees: Int = 0): Triple<ByteArray, Int, Int> {
        val width = imageProxy.width
        val height = imageProxy.height

        // Use CameraX's built-in toBitmap() which handles YUV_420_888 correctly
        var bitmap = imageProxy.toBitmap()
        d("[DEBUG] imageProxy size: ${width}x${height}")
        d("[DEBUG] Before rotation bitmap: ${bitmap.width}x${bitmap.height}, correction: $rotationDegrees")

        // Apply rotation if needed
        if (rotationDegrees != 0) {
            bitmap = rotateBitmap(bitmap, rotationDegrees)
        }

        // Get actual dimensions after rotation
        val finalWidth = bitmap.width
        val finalHeight = bitmap.height
        d("[DEBUG] After rotation bitmap: ${finalWidth}x${finalHeight}")

        // Convert bitmap to RGB byte array
        val rgbBytes = ByteArray(finalWidth * finalHeight * 3)
        val pixels = IntArray(finalWidth * finalHeight)
        bitmap.getPixels(pixels, 0, finalWidth, 0, 0, finalWidth, finalHeight)
        bitmap.recycle()

        var i = 0
        for (pixel in pixels) {
            rgbBytes[i++] = ((pixel shr 16) and 0xFF).toByte()  // R
            rgbBytes[i++] = ((pixel shr 8) and 0xFF).toByte()   // G
            rgbBytes[i++] = (pixel and 0xFF).toByte()           // B
        }

        d("[DEBUG] Final RGB bytes: ${rgbBytes.size} bytes, size: ${finalWidth}x${finalHeight}")
        return Triple(rgbBytes, finalWidth, finalHeight)
    }

    /**
     * Rotate a bitmap by the specified degrees.
     */
    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap

        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees.toFloat())

        return android.graphics.Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    /**
     * Scale RGB byte array to a smaller size.
     * Uses OpenCV's pyrDown() for true Gaussian pyramid downsampling.
     * This applies proper 5-tap Gaussian pre-filtering to reduce aliasing artifacts.
     *
     * @param rgbBytes RGB byte array (width * height * 3 bytes)
     * @param width Original width
     * @param height Original height
     * @param scale Scale factor (0.1 to 1.0)
     * @return Triple of (scaled RGB bytes, scaled width, scaled height)
     */
    fun scaleRgbBytes(rgbBytes: ByteArray, width: Int, height: Int, scale: Float): Triple<ByteArray, Int, Int> {
        val scaledWidth = (width * scale).toInt().coerceAtLeast(1)
        val scaledHeight = (height * scale).toInt().coerceAtLeast(1)

        // Handle scale=1.0 case: no scaling needed
        if (scaledWidth == width && scaledHeight == height) {
            return Triple(rgbBytes, width, height)
        }

        // Always use OpenCV pyrDown for true Gaussian pyramid downsampling
        // OpenCV is initialized in VlmApplication.onCreate()
        return scaleWithGaussianPyramid(rgbBytes, width, height, scaledWidth, scaledHeight)
    }

    /**
     * Gaussian pyramid downsampling using OpenCV pyrDown().
     * Applies proper 5-tap Gaussian filter before each 2x downsampling step.
     */
    private fun scaleWithGaussianPyramid(rgbBytes: ByteArray, width: Int, height: Int, targetWidth: Int, targetHeight: Int): Triple<ByteArray, Int, Int> {
        // Convert RGB bytes to OpenCV Mat (CV_8UC3)
        val srcMat = ByteArrayToMat(rgbBytes, width, height)

        var currentMat = srcMat
        var currentWidth = width
        var currentHeight = height

        // Apply pyrDown repeatedly until we reach or pass target size
        // Each pyrDown halves the dimensions with Gaussian pre-filtering
        while (currentWidth > targetWidth * 1.5f && currentHeight > targetHeight * 1.5f) {
            val nextWidth = (currentWidth / 2).coerceAtLeast(targetWidth)
            val nextHeight = (currentHeight / 2).coerceAtLeast(targetHeight)

            val nextMat = Mat()
            Imgproc.pyrDown(currentMat, nextMat, Size(nextWidth.toDouble(), nextHeight.toDouble()))

            if (currentMat != srcMat) {
                currentMat.release()
            }
            currentMat = nextMat
            currentWidth = nextWidth
            currentHeight = nextHeight
        }

        // Final bilinear scale to exact target if needed
        val resultMat = if (currentWidth != targetWidth || currentHeight != targetHeight) {
            val finalMat = Mat()
            Imgproc.resize(currentMat, finalMat, Size(targetWidth.toDouble(), targetHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            if (currentMat != srcMat) {
                currentMat.release()
            }
            finalMat
        } else {
            currentMat
        }

        // Convert back to RGB bytes
        val result = MatToRgbBytes(resultMat)
        resultMat.release()

        return Triple(result, targetWidth, targetHeight)
    }

    /**
     * Convert RGB byte array to OpenCV Mat (CV_8UC3).
     */
    private fun ByteArrayToMat(rgbBytes: ByteArray, width: Int, height: Int): Mat {
        val mat = Mat(height, width, CvType.CV_8UC3)
        val bytes = rgbBytes
        mat.put(0, 0, bytes)
        return mat
    }

    /**
     * Convert OpenCV Mat (CV_8UC3) to RGB byte array.
     */
    private fun MatToRgbBytes(mat: Mat): ByteArray {
        val bytes = ByteArray(mat.total().toInt() * mat.channels())
        mat.get(0, 0, bytes)
        return bytes
    }

    /**
     * Convert RGB byte array to JPEG base64 string.
     * Used for REMOTE mode inference.
     *
     * @param rgbBytes RGB byte array (width * height * 3 bytes)
     * @param width Image width
     * @param height Image height
     * @param quality JPEG quality (0-100)
     * @return Base64 encoded JPEG string
     */
    fun rgbBytesToJpegBase64(rgbBytes: ByteArray, width: Int, height: Int, quality: Int = 100): String {
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
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        bitmap.recycle()

        return android.util.Base64.encodeToString(outputStream.toByteArray(), android.util.Base64.NO_WRAP)
    }
}
