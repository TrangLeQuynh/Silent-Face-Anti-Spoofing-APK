package com.mv.livebodyexample

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.util.Rational
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.mv.livebodyexample.databinding.ActivityMainBinding
import kotlinx.coroutines.ObsoleteCoroutinesApi
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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
    private val cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

    //AspectRatio.RATIO_4_3 : 640 / 480
    private val previewWidth: Int = 640
    private val previewHeight: Int = 480
    private var screenWidth: Int = 0
    private var screenHeight: Int = 0
    private var factorX: Float = 0F
    private var factorY: Float = 0F

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        binding.result = DetectionResult()
        setContentView(binding.root)
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
                bindCameraUseCases()
            } else {
                Toast.makeText(this, R.string.permission_deny_camera, Toast.LENGTH_LONG).show()
            }
        }
    }

    @SuppressLint("RestrictedApi")
    private fun bindCameraUseCases() = binding.viewFinder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            // Preview
            preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(binding.viewFinder.display.rotation)
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
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        DetectionAnalyzer(baseContext, cameraSelector, binding.viewFinder, binding.faceImageView) { result ->
//                            result.updateLocation(calculateBoxLocationOnScreen(Rect(120, 160, 360, 480)))
                            binding.result = result
                            binding.rectView.postInvalidate()
                        })
                }


            try {

                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                //config viewPort
                val viewPort = ViewPort.Builder(
                    Rational(binding.viewFinder.width, binding.viewFinder.height),
                    binding.viewFinder.display.rotation
                ).build()
                Log.d(TAG, "View port: ${binding.viewFinder.width} x ${binding.viewFinder.height}")

                val useCaseGroup = UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(imageAnalyzer)
                    .setViewPort(viewPort)
                    .build()

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    useCaseGroup
                )

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

                // Attach the preview to preview view, aka View Finder
                preview.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onResume() {
//        engineWrapper = EngineWrapper(assets)
//        enginePrepared = engineWrapper.init()
//        if (!enginePrepared) {
//            Toast.makeText(this, "Engine init failed.", Toast.LENGTH_LONG).show()
//        }
//        Log.d(TAG_ENGINE_WRAPPER, "Engine init $enginePrepared")
        if (allPermissionsGranted()) {
            bindCameraUseCases()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSION_REQUEST_CODE)
        }
        super.onResume()
    }

    override fun onDestroy() {
        engineWrapper.destroy()
        cameraExecutor.shutdown()
        super.onDestroy()
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
