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

import lombok.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
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
@Data
public class FeedForwardToCnnPreProcessor implements InputPreProcessor {
    private int inputHeight;
    private int inputWidth;
    private int numChannels;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int[] shape;

    /**
     * Reshape to a channels x rows x columns tensor
     * @param inputHeight the columns
     * @param inputWidth the rows
     * @param numChannels the channels
     */
    @JsonCreator
    public FeedForwardToCnnPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                                        @JsonProperty("inputWidth") int inputWidth,
                                        @JsonProperty("numChannels") int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    public FeedForwardToCnnPreProcessor(int inputHeight, int inputWidth) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    @Override
    public INDArray preProcess(INDArray input, Layer layer) {
        this.shape = input.shape();
        if(input.shape().length == 4)
            return input;
        if(input.columns() != inputWidth * inputHeight)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputHeight + " x columns " + inputWidth  + " but was instead " + Arrays.toString(input.shape()));
        return input.reshape(input.size(0),numChannels,inputHeight,inputWidth);
    }

    @Override
    // return 4 dimensions
    public INDArray backprop(INDArray output, Layer layer){
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
            System.arraycopy(output.shape(), 1, otherOutputs, 0, otherOutputs.length);
            shape = new int[] {output.shape()[0], ArrayUtil.prod(otherOutputs)};
        }
        return output.reshape(shape);
    }

    @Override
    public FeedForwardToCnnPreProcessor clone() {
        try {
            FeedForwardToCnnPreProcessor clone = (FeedForwardToCnnPreProcessor) super.clone();
            if(clone.shape != null) clone.shape = clone.shape.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
