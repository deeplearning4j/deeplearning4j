/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.objdetect;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A detected object, by an object detection algorithm.
 * Note that the dimensions (for center X/Y, width/height) depend on the specific implementation.
 * For example, in the {@link Yolo2OutputLayer}, the dimensions are grid cell units - for example, with 416x416 input,
 * 32x downsampling, we have 13x13 grid cells (each corresponding to 32 pixels in the input image). Thus, a centerX
 * of 5.5 would be xPixels=5.5x32 = 176 pixels from left. Widths and heights are similar: in this example, a with of 13
 * would be the entire image (416 pixels), and a height of 6.5 would be 6.5/13 = 0.5 of the image (208 pixels).
 *
 * @author Alex Black
 */
@Data
public class DetectedObject {

    private final int exampleNumber;
    private final double centerX;
    private final double centerY;
    private final double width;
    private final double height;
    private final INDArray classPredictions;
    private int predictedClass = -1;
    private final double confidence;


    /**
     * @param exampleNumber    Index of the example in the current minibatch. For single images, this is always 0
     * @param centerX          Center X position of the detected object
     * @param centerY          Center Y position of the detected object
     * @param width            Width of the detected object
     * @param height           Height of  the detected object
     * @param classPredictions Row vector of class probabilities for the detected object
     */
    public DetectedObject(int exampleNumber, double centerX, double centerY, double width, double height,
                          INDArray classPredictions, double confidence){
        this.exampleNumber = exampleNumber;
        this.centerX = centerX;
        this.centerY = centerY;
        this.width = width;
        this.height = height;
        this.classPredictions = classPredictions;
        this.confidence = confidence;
    }

    /**
     * Get the top left X/Y coordinates of the detected object
     *
     * @return Array of length 2 - top left X and Y
     */
    public double[] getTopLeftXY(){
        return new double[]{ centerX - width / 2.0, centerY - height / 2.0};
    }

    /**
     * Get the bottom right X/Y coordinates of the detected object
     *
     * @return Array of length 2 - bottom right X and Y
     */
    public double[] getBottomRightXY(){
        return new double[]{ centerX + width / 2.0, centerY + height / 2.0};
    }

    /**
     * Get the index of the predicted class (based on maximum predicted probability)
     * @return Index of the predicted class (0 to nClasses - 1)
     */
    public int getPredictedClass(){
        if(predictedClass == -1){
            // ravel in case we get a column vector
            predictedClass = classPredictions.ravel().argMax(1).getInt(0);
        }
        return predictedClass;
    }

    public String toString() {
        return "DetectedObject(exampleNumber=" + exampleNumber + ", centerX=" + centerX + ", centerY=" + centerY +
                ", width=" + width + ", height=" + height + ", confidence=" + confidence
                + ", classPredictions=" + classPredictions + ", predictedClass=" + getPredictedClass() + ")";
    }
}
