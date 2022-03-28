package com.mv.livebodyexample

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.media.Image
import android.util.Log
import android.widget.ImageView
import androidx.annotation.RequiresApi
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayOutputStream
import java.lang.IllegalArgumentException
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage

typealias DetectionListener = (detector: DetectionResult) -> Unit

@RequiresApi(21)
class DetectionAnalyzer(
  ctx: Context,
  cameraSelector: CameraSelector,
  viewFinder: PreviewView,
  imageView: ImageView,
  private val listener: DetectionListener
) :  ImageAnalysis.Analyzer {

  private val TAG: String = "DetectionAnalyzer"
  private val linePaint: Paint = Paint().apply {
    color = Color.YELLOW
    style = Paint.Style.STROKE
    strokeWidth = 2f
  }

  private val dotPaint: Paint = Paint().apply {
    color = Color.RED
    style = Paint.Style.FILL
  }

  private val borderPaint: Paint = Paint().apply {
    color = Color.RED
    style = Paint.Style.STROKE
    strokeWidth = 6f
  }

  private var viewFinder: PreviewView
  private var imageView: ImageView
  private var cameraSelector: CameraSelector

  private lateinit var bitmap: Bitmap
  private lateinit var canvas: Canvas
  private var factorX = 1.0f
  private var factorY = 1.0f

  init {
    this.viewFinder = viewFinder
    this.cameraSelector = cameraSelector
    this.imageView = imageView
  }

  private val faceDetector: FaceDetector by lazy {
    val options = FaceDetectorOptions.Builder()
      .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
      .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
      .build()
    FaceDetection.getClient(options)
  }

  @SuppressLint("UnsafeOptInUsageError")
  override fun analyze(imageProxy: ImageProxy) {
    val image = imageProxy.image

    if (image == null) {
      imageProxy.close()
      Log.d(MainActivity.TAG_TF_LITE, "$TAG: image null")
      return
    }

    initDrawing(imageProxy.width, imageProxy.height)
    val input = InputImage.fromBitmap(image!!.toBitmap(), 0)
    val begin = System.currentTimeMillis()

    faceDetector.process(input)
      .addOnSuccessListener { faces ->
        //process face
        drawFaces(faces)
      }
      .addOnFailureListener { e ->
        Log.d(TAG, "$TAG: Fail: ${e.toString()}")
      }
    imageProxy.close()
  }

  private fun drawFaces(faces: List<Face>) {
    Log.d(TAG, "$TAG: Faces size ${faces.size}")
    if (faces.isEmpty()) {
      imageView.setImageBitmap(bitmap)
      return
    }
    for (face in faces) {
      for (landmark in face.allLandmarks) {
        canvas.drawCircle(
          landmark.position.x,
          landmark.position.y,
          5f,
          dotPaint
        )
      }
      val bounds = face.boundingBox
      val rotY = face.headEulerAngleY // Head is rotated to the right rotY degrees
      val rotZ = face.headEulerAngleZ // Head is tilted sideways rotZ degrees
      Log.d(TAG, "$TAG $bounds || $rotY || $rotZ")
      canvas.drawRect(bounds, linePaint)
    }
    imageView.setImageBitmap(bitmap)
  }

  private fun initDrawing(width: Int, height: Int) {
    bitmap = Bitmap.createBitmap(viewFinder.width, viewFinder.height, Bitmap.Config.ARGB_8888)
    canvas = Canvas(bitmap)

    canvas.drawRect(Rect(0, 0, bitmap.width, bitmap.height), borderPaint)

    factorX = canvas.width / (width * 1.0f)
    factorY = canvas.height / (height * 1.0f)
    Log.d(MainActivity.TAG_TF_LITE, "$TAG: Factorx= $factorX || factorY = $factorY")
  }

  private fun translateBoundingBox(box: Rect) : Rect {
    return Rect(
      translateX(box.left).toInt(),
      translateY(box.top).toInt(),
      translateX(box.right).toInt(),
      translateY(box.bottom).toInt()
    )
  }

  private fun translateX(x: Int) : Float {
    val scaledX = x * factorX
    if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) {
      return canvas.width - scaledX
    }
    return scaledX
  }

  private fun translateY(y: Int) : Float {
    return y * factorY
  }

  private fun degreeToRotation(degrees: Int) : Int {
    when(degrees) {
      0 -> return 0
      90 -> return 1
      180 -> return 2
      270 -> return 3
      else -> throw IllegalArgumentException("Rotation must be 0, 90, 180, or 270.")
    }
  }

  fun Image.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer // Y
    val vuBuffer = planes[2].buffer // VU

    val ySize = yBuffer.remaining()
    val vuSize = vuBuffer.remaining()

    val nv21 = ByteArray(ySize + vuSize)

    yBuffer.get(nv21, 0, ySize)
    vuBuffer.get(nv21, ySize, vuSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
  }
}

//private class DetectionAnalyzer(
//  ctx: Context,
//  private val listener: DetectionListener,
//) : ImageAnalysis.Analyzer {
//
//  // TODO 1: Add class variable TensorFlow Lite Model
//  // Initializing the model by lazy so that it runs in the same thread when the process
//  // method is called.
//  private val objectDetection: ObjectDetection by lazy {
//
//    // TODO 6. Optional GPU acceleration
//    val compatList = CompatibilityList()
//
//    val options = if(compatList.isDelegateSupportedOnThisDevice) {
//      Log.d(MainActivity.TAG, "This device is GPU Compatible ")
//      Model.Options.Builder().setDevice(Model.Device.GPU).build()
//    } else {
//      Log.d(MainActivity.TAG, "This device is GPU Incompatible ")
//      Model.Options.Builder().setNumThreads(4).build()
//    }
//
//    // Initialize the Flower Model
//    ObjectDetection.newInstance(ctx, options)
//  }
//
//  private fun ByteBuffer.toByteArray(): ByteArray {
//    rewind()    // Rewind the buffer to zero
//    val data = ByteArray(remaining())
//    get(data)   // Copy the buffer into a byte array
//    return data // Return the byte array
//  }
//
//  fun Image.toBitmap(): Bitmap {
//    val yBuffer = planes[0].buffer // Y
//    val vuBuffer = planes[2].buffer // VU
//
//    val ySize = yBuffer.remaining()
//    val vuSize = vuBuffer.remaining()
//
//    val nv21 = ByteArray(ySize + vuSize)
//
//    yBuffer.get(nv21, 0, ySize)
//    vuBuffer.get(nv21, ySize, vuSize)
//
//    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
//    val out = ByteArrayOutputStream()
//    yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
//    val imageBytes = out.toByteArray()
//    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//  }
//
//  private var working: Boolean = false;
//
//  @SuppressLint("UnsafeOptInUsageError")
//  override fun analyze(imageProxy: ImageProxy) {
//    val image = imageProxy.image
//    if (image == null) {
//      imageProxy.close()
//      Log.d(MainActivity.TAG_TF_LITE, "${MainActivity.TAG_TF_LITE} image null")
//    }
//    if (!working) {
//      Log.d(MainActivity.TAG_TF_LITE, "Image cropRect ${imageProxy.cropRect}")
//      Log.d(MainActivity.TAG_TF_LITE, "Image size w-h: ${imageProxy.width} ${imageProxy.height}")
//      Log.d(MainActivity.TAG_TF_LITE, "Image info: ${imageProxy.imageInfo}")
//      Log.d(MainActivity.TAG_TF_LITE, "Image format: ${imageProxy.format}")
//      working = true
//    }
//
//    // TODO 2: Convert Image to Bitmap then to TensorImage
//    val tfImage = TensorImage.fromBitmap(image!!.toBitmap())
//
//    // TODO 3: Process the image using the trained model, sort and pick out the top results
//    // Runs model inference and gets result.
//    val outputs = objectDetection.process(tfImage).detectionResultList
//      .apply {
//        sortByDescending {
//          it.scoreAsFloat
//        }
//      }
//    // TODO 4: Converting the top probability items into a list of recognitions
//    val detectionResult = outputs.get(0)
//
//    // Gets result from DetectionResult.
//    val location = detectionResult.locationAsRectF;
//    val category = detectionResult.categoryAsString;
//    val score = detectionResult.scoreAsFloat;
////            Log.d(TAG_TF_LITE, "Image $category")
//
//    val rect = Rect(
//      location.left.toInt(),
//      location.top.toInt(),
//      location.right.toInt(),
//      location.bottom.toInt()
//    )
//
//    val result = DetectionResult(rect, score, 0, true)
//    listener(result)
//
//    imageProxy.close()
//  }
//}