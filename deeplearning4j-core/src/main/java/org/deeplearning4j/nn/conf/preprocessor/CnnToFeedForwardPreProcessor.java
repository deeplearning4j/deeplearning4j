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
        if(input.rank() == 2) return input; //Should never happen

        //Assume input is standard rank 4 activations out of CNN layer
        //First: we require input to be in c order. But c order (as declared in array order) isn't enough; also need strides to be correct
        if(input.ordering() != 'c' || !Shape.strideDescendingCAscendingF(input)) input = input.dup('c');

        int[] inShape = input.shape();  //[miniBatch,depthOut,outH,outW]
        int[] outShape = new int[]{inShape[0], inShape[1]*inShape[2]*inShape[3]};

        return input.reshape('c',outShape);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize){
        //Epsilons from layer above should be 2d, with shape [miniBatchSize, depthOut*outH*outW]
        if(epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons)) epsilons = epsilons.dup('c');

        if(epsilons.rank() == 4) return epsilons;   //Should never happen

        if(epsilons.columns() != inputWidth * inputHeight * numChannels )
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputHeight
                    + " x columns " + inputWidth + " x depth " + numChannels +" but was instead " + Arrays.toString(epsilons.shape()));

        return epsilons.reshape('c', epsilons.size(0), numChannels, inputHeight, inputWidth);
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
