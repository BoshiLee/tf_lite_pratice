package com.example.test_yolo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private var mImage: ImageView? = null
    private var mText: TextView? = null
    private var mbtn: Button? = null
    private var objectDetectorHelper: ObjectDetectorHelper = ObjectDetectorHelper(this)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        mImage = findViewById(R.id.assetsImageView)
        mImage?.setBackgroundColor(Color.rgb(0, 0, 0))
        mText = findViewById(R.id.inferenceTextView)
        mbtn = findViewById(R.id.readImgBtn)

        // Setup digit classifier.
        objectDetectorHelper
            .initialize()
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up yolov5 model.", e) }
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
        val imageRotation = 0
        // Pass Bitmap and rotation to the object detector helper for processing and detection
//        objectDetectorHelper.detect(image, imageRotation)
    }

    companion object {
        private const val TAG = "MainActivity"
    }

}

