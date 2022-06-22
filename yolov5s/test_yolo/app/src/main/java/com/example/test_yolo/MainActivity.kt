package com.example.test_yolo

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import java.io.IOException
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.task.vision.detector.Detection

class MainActivity : AppCompatActivity(), ObjectDetectorHelper.DetectorListener {

    private var mImage: ImageView? = null
    private var mText: TextView? = null
    private var mbtn: Button? = null
    private lateinit var bitmapBuffer: Bitmap
    private val TAG = "ObjectDetection"
    private lateinit var objectDetectorHelper: ObjectDetectorHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        mImage = findViewById(R.id.assetsImageView)
        mImage?.setBackgroundColor(Color.rgb(0, 0, 0))
        mText = findViewById(R.id.inferenceTextView)
        mbtn = findViewById(R.id.readImgBtn)

    }

    fun btnReadImageClick(view: View) {
        this.loadDataFromAsset()
        mText?.text= "Button is Clicked!"
    }

    fun loadDataFromAsset() {
        try {
            val bitmap = assets.open("bus.jpg")
            val bit = BitmapFactory.decodeStream(bitmap)
            mImage?.setImageBitmap(bit)

            detectObjects(bit)
        } catch (e1: IOException) {
            // TODO Auto-generated catch block
            e1.printStackTrace()
        }
    }

    private fun detectObjects(bitmap: Bitmap) {

        // Copy out RGB bits to the shared bitmap buffer
        val image: Bitmap = Bitmap.createBitmap(
            bitmap.width,
            bitmap.height,
            Bitmap.Config.ARGB_8888
        )

        val imageRotation = 0        // Pass Bitmap and rotation to the object detector helper for processing and detection
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }

    override fun onError(error: String) {
        Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
    }

    override fun onResults(results: MutableList<Detection>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int) {
        this.runOnUiThread {
            mText?.text = String.format("%d ms", inferenceTime)


            // Pass necessary information to OverlayView for drawing on the canvas
//            this.overlay.setResults(
//                results ?: LinkedList<Detection>(),
//                imageHeight,
//                imageWidth
//            )

            // Force a redraw
//            this.overlay.invalidate()
        }
    }


}

