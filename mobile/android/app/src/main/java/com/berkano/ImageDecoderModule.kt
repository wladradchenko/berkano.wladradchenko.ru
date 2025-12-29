package com.berkano;

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.facebook.react.bridge.*
import java.io.InputStream

class ImageDecoderModule(
  private val reactContext: ReactApplicationContext
) : ReactContextBaseJavaModule(reactContext) {

  override fun getName(): String {
    return "ImageDecoder"
  }

  @ReactMethod
  fun decodeToTensor(
    uriString: String,
    targetWidth: Int,
    targetHeight: Int,
    promise: Promise
  ) {
    try {
      val uri = Uri.parse(uriString)
      val inputStream: InputStream? =
        reactContext.contentResolver.openInputStream(uri)

      if (inputStream == null) {
        promise.reject("ERROR", "Cannot open image stream")
        return
      }

      var bitmap: Bitmap? = null
      try {
        bitmap = BitmapFactory.decodeStream(inputStream)
        if (bitmap == null) {
          promise.reject("ERROR", "Cannot decode image")
          return
        }
        bitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
      } finally {
        inputStream.close()
      }

      val width = bitmap.width
      val height = bitmap.height

      val pixels = IntArray(width * height)
      bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

      // CHW: [3, H, W] - формат для передачи в JavaScript
      val tensor = FloatArray(3 * width * height)

      for (y in 0 until height) {
        for (x in 0 until width) {
          val color = pixels[y * width + x]

          val r = (color shr 16 and 0xFF) / 255f
          val g = (color shr 8 and 0xFF) / 255f
          val b = (color and 0xFF) / 255f

          val idx = y * width + x
          tensor[idx] = r
          tensor[width * height + idx] = g
          tensor[2 * width * height + idx] = b
        }
      }

      // Конвертируем FloatArray в WritableArray для правильной передачи в JavaScript
      // React Native не может напрямую конвертировать FloatArray, поэтому используем WritableArray
      val writableArray = Arguments.createArray()
      for (value in tensor) {
        // pushDouble принимает double, поэтому конвертируем float в double
        writableArray.pushDouble(value.toDouble())
      }
      
      promise.resolve(writableArray)
    } catch (e: Exception) {
      promise.reject("ERROR_DECODING_IMAGE", e)
    }
  }
}

