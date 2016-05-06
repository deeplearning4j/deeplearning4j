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
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 *
 *
 * A preprocessor to allow CNN and standard feed-forward network layers to be used together.<br>
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
    private int inputHeight;
    private int inputWidth;
    private int numChannels;

    /**
     * @param inputHeight the columns
     * @param inputWidth the rows
     * @param numChannels the channels
     */

    @JsonCreator
    public CnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                                        @JsonProperty("inputWidth") int inputWidth,
                                        @JsonProperty("numChannels") int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    public CnnToFeedForwardPreProcessor(int inputHeight, int inputWidth) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    public CnnToFeedForwardPreProcessor(){}

    @Override
    // return 2 dimensions
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        int[] otherOutputs = null;

        //this.inputHeight = input.size(-2);
        //this.inputWidth = input.size(-1);

        if(input.shape().length == 2) {
            return input;
        }
        else if(input.shape().length == 4) {
            if(input.size(-2) == 1 && input.size(-1) == 1) {
                return input.reshape(input.size(0), input.size(1));
            }
            //this.numChannels = input.size(-3);
            otherOutputs = new int[3];
        }
        else if(input.shape().length == 3) {
            otherOutputs = new int[2];
        }
        System.arraycopy(input.shape(), 1, otherOutputs, 0, otherOutputs.length);
        int[] shape = new int[] {input.size(0), ArrayUtil.prod(otherOutputs)};
        if(input.ordering() == 'f') input = Shape.toOffsetZeroCopy(input,'c');
        return input.reshape(shape);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize){
        if (output.shape().length == 4)
            return output;
        if (output.columns() != inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputHeight
                    + " x columns " + inputWidth + " x depth " + numChannels +" but was instead " + Arrays.toString(output.shape()));
        if(output.ordering() == 'f') output = Shape.toOffsetZeroCopy(output,'c');
        return output.reshape(output.size(0), numChannels, inputHeight, inputWidth);
    }

    @Override
    public CnnToFeedForwardPreProcessor clone() {
        try {
            CnnToFeedForwardPreProcessor clone = (CnnToFeedForwardPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
