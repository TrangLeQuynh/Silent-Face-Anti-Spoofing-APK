package com.mv.livebodyexample

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.mv.livebodyexample.databinding.ActivityMainBinding
import com.mv.livebodyexample.ml.ObjectDetection
import kotlinx.coroutines.ObsoleteCoroutinesApi
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.model.Model
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

typealias DetectionListener = (detector: DetectionResult) -> Unit

@ObsoleteCoroutinesApi
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private var enginePrepared: Boolean = false
    private lateinit var engineWrapper: EngineWrapper

    /// Camerax variables
    private lateinit var preview: Preview // Preview use case, fast, responsive view of the camera
    private lateinit var imageAnalyzer: ImageAnalysis // Analysis use case, for running ML code
    private lateinit var camera: Camera
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    //AspectRatio.RATIO_4_3 : 640 / 480
    private val previewWidth: Int = 640
    private val previewHeight: Int = 480
    private var screenWidth: Int = 0
    private var screenHeight: Int = 0
    private var factorX: Float = 0F
    private var factorY: Float = 0F

    // Views attachment
    private val viewFinder by lazy {
        findViewById<PreviewView>(R.id.view_finder)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        binding.result = DetectionResult()
        setContentView(binding.root)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSION_REQUEST_CODE)
        }
    }

    private fun allPermissionsGranted(): Boolean = PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    @SuppressLint("MissingSuperCall")
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, R.string.permission_deny_camera, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun startCamera() {
        calculateSize()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            // Preview
            preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            //analyzer
            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(binding.viewFinder.display.rotation)
                //This guarantees only one image will be delivered for analysis at a time
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, DetectionAnalyzer(baseContext) { result ->
                        result.updateLocation(calculateBoxLocationOnScreen(Rect(120, 160, 360, 480)))
                        binding.result = result
                        binding.rectView.postInvalidate()
                    })
                }


            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

                // Attach the preview to preview view, aka View Finder
                preview.setSurfaceProvider(viewFinder.surfaceProvider)
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun calculateBoxLocationOnScreen(rect: Rect): Rect =
        Rect(
            (rect.left * factorX).toInt(),
            (rect.top * factorY).toInt(),
            (rect.right * factorX).toInt(),
            (rect.bottom * factorY).toInt()
        )

    private fun calculateSize() {
        val dm = DisplayMetrics()
        windowManager.defaultDisplay.getMetrics(dm)
        screenWidth = dm.widthPixels
        screenHeight = dm.heightPixels

        factorX = screenWidth / previewHeight.toFloat()
        factorY = screenHeight / previewWidth.toFloat()

        Log.d(TAG, "Image: Screen size $screenWidth x $screenHeight")
        Log.d(TAG, "Image: View Finder size ${binding.viewFinder.width} x ${binding.viewFinder.height}")
    }

    override fun onResume() {
        engineWrapper = EngineWrapper(assets)
        enginePrepared = engineWrapper.init()
        if (!enginePrepared) {
            Toast.makeText(this, "Engine init failed.", Toast.LENGTH_LONG).show()
        }
        Log.d(TAG_ENGINE_WRAPPER, "Engine init $enginePrepared")
        super.onResume()
    }

    override fun onDestroy() {
        engineWrapper.destroy()
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    private class DetectionAnalyzer(
        ctx: Context,
        private val listener: DetectionListener,
    ) : ImageAnalysis.Analyzer {

        // TODO 1: Add class variable TensorFlow Lite Model
        // Initializing the model by lazy so that it runs in the same thread when the process
        // method is called.
        private val objectDetection: ObjectDetection by lazy {

            // TODO 6. Optional GPU acceleration
            val compatList = CompatibilityList()

            val options = if(compatList.isDelegateSupportedOnThisDevice) {
                Log.d(TAG, "This device is GPU Compatible ")
                Model.Options.Builder().setDevice(Model.Device.GPU).build()
            } else {
                Log.d(TAG, "This device is GPU Incompatible ")
                Model.Options.Builder().setNumThreads(4).build()
            }

            // Initialize the Flower Model
            ObjectDetection.newInstance(ctx, options)
        }

        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()    // Rewind the buffer to zero
            val data = ByteArray(remaining())
            get(data)   // Copy the buffer into a byte array
            return data // Return the byte array
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

        private var working: Boolean = false;

        @SuppressLint("UnsafeOptInUsageError")
        override fun analyze(imageProxy: ImageProxy) {
            val image = imageProxy.image
            if (image == null) {
                imageProxy.close()
                Log.d(TAG_TF_LITE, "$TAG_TF_LITE image null")
            }
            if (!working) {
                Log.d(TAG_TF_LITE, "Image cropRect ${imageProxy.cropRect}")
                Log.d(TAG_TF_LITE, "Image size w-h: ${imageProxy.width} ${imageProxy.height}")
                Log.d(TAG_TF_LITE, "Image info: ${imageProxy.imageInfo}")
                Log.d(TAG_TF_LITE, "Image format: ${imageProxy.format}")
                working = true
            }

            // TODO 2: Convert Image to Bitmap then to TensorImage
            val tfImage = TensorImage.fromBitmap(image!!.toBitmap())

            // TODO 3: Process the image using the trained model, sort and pick out the top results
            // Runs model inference and gets result.
            val outputs = objectDetection.process(tfImage).detectionResultList
                .apply {
                    sortByDescending {
                        it.scoreAsFloat
                    }
                }
            // TODO 4: Converting the top probability items into a list of recognitions
            val detectionResult = outputs.get(0)

            // Gets result from DetectionResult.
            val location = detectionResult.locationAsRectF;
            val category = detectionResult.categoryAsString;
            val score = detectionResult.scoreAsFloat;
//            Log.d(TAG_TF_LITE, "Image $category")

            val rect = Rect(
                location.left.toInt(),
                location.top.toInt(),
                location.right.toInt(),
                location.bottom.toInt()
            )

            val result = DetectionResult(rect, score, 0, true)
            listener(result)

            imageProxy.close()
        }
    }

    companion object {
        const val TAG = "MainActivity"
        const val TAG_CAMERAX = "Camera-X"
        const val TAG_TF_LITE = "TF-Lite"
        const val TAG_ENGINE_WRAPPER = "Engine Wrapper"

        const val DEFAULT_THRESHOLD = 0.915F

        val PERMISSIONS: Array<String> = arrayOf(Manifest.permission.CAMERA)
        const val PERMISSION_REQUEST_CODE = 1
    }
}
