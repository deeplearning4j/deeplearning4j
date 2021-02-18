/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.objdetect;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

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
            if(classPredictions.rank() == 1){
                predictedClass = classPredictions.argMax().getInt(0);
            } else {
                // ravel in case we get a column vector, or rank 2 row vector, etc
                predictedClass = classPredictions.ravel().argMax().getInt(0);
            }
        }
        return predictedClass;
    }

    public String toString() {
        return "DetectedObject(exampleNumber=" + exampleNumber + ", centerX=" + centerX + ", centerY=" + centerY +
                ", width=" + width + ", height=" + height + ", confidence=" + confidence
                + ", classPredictions=" + classPredictions + ", predictedClass=" + getPredictedClass() + ")";
    }
}
