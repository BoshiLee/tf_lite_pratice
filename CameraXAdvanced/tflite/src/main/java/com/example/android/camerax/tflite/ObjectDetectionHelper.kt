/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camerax.tflite

import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer

/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class ObjectDetectionHelper(private val tflite: Interpreter, private val labels: List<String>) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(val location: RectF, val label: String, val score: Float)

    private val locations = arrayOf(Array(OBJECT_COUNT) { FloatArray(4) })
    private val labelIndices =  arrayOf(FloatArray(OBJECT_COUNT))
    private val scores =  arrayOf(FloatArray(OBJECT_COUNT))

    // Float model
    private val IMAGE_MEAN = 0f

    private val IMAGE_STD = 255.0f

    private val outputBuffer = mapOf(
        0 to locations,
        1 to labelIndices,
        2 to scores,
        3 to FloatArray(1)
    )

// 將模型輸出的結果處理成 ObjectPrediction 列表
private val predictions get() = (0 until OBJECT_COUNT).map { count ->
    ObjectPrediction(

        // The locations are an array of [0, 1] floats for [top, left, bottom, right]
        location = locations[0][count].let { locat ->
            RectF(locat[1], locat[0], locat[3], locat[2])
        },

        // SSD Mobilenet V1 Model assumes class 0 is background class
        // in label file and class labels start from 1 to number_of_classes + 1,
        // while outputClasses correspond to class index from 0 to number_of_classes
        label = labels[1 + labelIndices[0][count].toInt()],

        // Score is a single value of [0, 1]
        score = scores[0][count]
    )
}

    fun predict(image: TensorImage): List<ObjectPrediction> {
        tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)
        return predictions
    }
//
//    protected fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer? {
////        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
////        byteBuffer.order(ByteOrder.nativeOrder());
////        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
//        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
//        val pixel = 0
//        imgData.rewind()
//        for (i in 0 until INPUT_SIZE) {
//            for (j in 0 until INPUT_SIZE) {
//                val pixelValue: Int = intValues.get(i * INPUT_SIZE + j)
//                if (isModelQuantized) {
//                    // Quantized model
//                    imgData.put((((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toByte())
//                    imgData.put((((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toByte())
//                    imgData.put((((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toByte())
//                } else { // Float model
//                    imgData.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
//                    imgData.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
//                    imgData.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
//                }
//            }
//        }
//        return imgData
//    }
//
//    fun recognizeImage(bitmap: Bitmap): List<ObjectPrediction> {
//        val byteBuffer_: ByteBuffer = convertBitmapToByteBuffer(bitmap)
//        tflite.runForMultipleInputsOutputs(arrayOf(bitmap.buffer), outputBuffer)
//        return predictions
//    }

    companion object {
        const val OBJECT_COUNT = 10
    }
}