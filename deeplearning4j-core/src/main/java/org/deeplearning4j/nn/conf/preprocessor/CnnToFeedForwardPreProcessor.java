/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.preprocessor;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

 /**A preprocessor to allow CNN and standard feed-forward network layers to be used together.<br>
 * For example, CNN -> Denselayer <br>
 * This does two things:<br>
 * (b) Reshapes 4d activations out of CNN layer, with shape
 * [numExamples, numChannels, inputHeight, inputWidth]) into 2d activations (with shape
 * [numExamples, inputHeight*inputWidth*numChannels]) for use in feed forward layer
 * (a) Reshapes epsilons (weights*deltas) out of FeedFoward layer (which is 2D or 3D with shape
 * [numExamples, inputHeight*inputWidth*numChannels]) into 4d epsilons (with shape
 * [numExamples, numChannels, inputHeight, inputWidth]) suitable to feed into CNN layers.<br>
 * Note: numChannels is equivalent to depth or featureMaps referenced in different literature
 * @author Adam Gibson
 * @see FeedForwardToCnnPreProcessor for opposite case (i.e., DenseLayer -> CNNetc)
*/
@Data
public class CnnToFeedForwardPreProcessor implements InputPreProcessor {
     private int inputWidth;
     private int inputHeight;
     private int numChannels;

     /**
      * @param inputWidth the rows
      * @param inputHeight the columns
      * @param numChannels the channels
      */

     @JsonCreator
    public CnnToFeedForwardPreProcessor(@JsonProperty("inputWidth") int inputWidth,
                                        @JsonProperty("inputHeight") int inputHeight,
                                        @JsonProperty("numChannels") int numChannels) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.numChannels = numChannels;
    }

    public CnnToFeedForwardPreProcessor(int inputWidth, int inputHeight) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.numChannels = 1;
    }

    public CnnToFeedForwardPreProcessor(){}

    @Override
    public INDArray preProcess(INDArray input) {
        int[] otherOutputs = null;

        if(input.shape().length == 2) {
            return input;
        } else if(input.shape().length == 4) {
            otherOutputs = new int[3];
        }
        else if(input.shape().length == 3) {
            otherOutputs = new int[2];
        }
        int[] shape = new int[] {input.shape()[0], ArrayUtil.prod(otherOutputs)};
        return input.reshape(shape);
    }

    @Override
    public INDArray backprop(INDArray output){
        if (output.shape().length == 4)
            return output;
        if (output.columns() != inputWidth * inputHeight)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputWidth + " x columns " + inputHeight + " but was instead " + Arrays.toString(output.shape()));
        return output.reshape(output.size(0), numChannels, inputWidth, inputHeight);
    }

}
