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

import lombok.EqualsAndHashCode;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**A preprocessor to allow CNN and standard feed-forward network layers to be used together.<br>
 * For example, DenseLayer -> CNN<br>
 * This does two things:<br>
 * (a) Reshapes activations out of FeedFoward layer (which is 2D or 3D with shape
 * [numExamples, inputHeight*inputWidth*numChannels]) into 4d activations (with shape
 * [numExamples, numChannels, inputHeight, inputWidth]) suitable to feed into CNN layers.<br>
 * (b) Reshapes 4d epsilons (weights*deltas) from CNN layer, with shape
 * [numExamples, numChannels, inputHeight, inputWidth]) into 2d epsilons (with shape
 * [numExamples, inputHeight*inputWidth*numChannels]) for use in feed forward layer
 * Note: numChannels is equivalent to depth or featureMaps referenced in different literature
 * @author Adam Gibson
 * @see CnnToFeedForwardPreProcessor for opposite case (i.e., CNN -> DenseLayer etc)

 */
@EqualsAndHashCode
public class FeedForwardToCnnPreProcessor implements InputPreProcessor {
    private int inputWidth;
    private int inputHeight;
    private int numChannels;
    private int[] shape;

    /**
     * Reshape to a channels x rows x columns tensor
     * @param inputWidth the rows
     * @param inputHeight the columns
     * @param numChannels the channels
     */
    public FeedForwardToCnnPreProcessor(int inputWidth, int inputHeight, int numChannels) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.numChannels = numChannels;
    }

    public FeedForwardToCnnPreProcessor(int inputWidth, int inputHeight) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.numChannels = 1;
    }

    public FeedForwardToCnnPreProcessor(){}

    @Override
    public INDArray preProcess(INDArray input) {
        this.shape = input.shape();
        if(input.shape().length == 4)
            return input;
        if(input.columns() != inputWidth * inputHeight)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputWidth + " x columns " + inputHeight + " but was instead " + Arrays.toString(input.shape()));
        return input.reshape(input.size(0),numChannels,inputWidth,inputHeight);
    }

    @Override
    public INDArray backprop(INDArray output){
        if(shape == null || ArrayUtil.prod(shape) != output.length()) {
            int[] otherOutputs = null;
            if(output.shape().length == 2) {
                return output;
            } else if(output.shape().length == 4) {
                otherOutputs = new int[3];
            }
            else if(output.shape().length == 3) {
                otherOutputs = new int[2];
            }
            shape = new int[] {output.shape()[0], ArrayUtil.prod(otherOutputs)};
        }
        return output.reshape(shape);
    }

}
