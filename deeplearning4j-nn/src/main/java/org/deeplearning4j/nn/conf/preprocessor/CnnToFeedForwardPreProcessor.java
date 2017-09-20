/*-
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

import lombok.Data;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

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
                    @JsonProperty("inputWidth") int inputWidth, @JsonProperty("numChannels") int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    public CnnToFeedForwardPreProcessor(int inputHeight, int inputWidth) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    public CnnToFeedForwardPreProcessor() {}

    @Override
    // return 2 dimensions
    public Activations preProcess(Activations activations, int miniBatchSize, boolean training) {
        if(activations.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess input: Activations must have exactly 1 array. Got: "
                    + activations.size());
        }
        INDArray input = activations.get(0);
        if (input.rank() == 2)
            return activations; //Should usually never happen

        //Assume input is standard rank 4 activations out of CNN layer
        //First: we require input to be in c order. But c order (as declared in array order) isn't enough; also need strides to be correct
        if (input.ordering() != 'c' || !Shape.strideDescendingCAscendingF(input))
            input = input.dup('c');

        int[] inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        int[] outShape = new int[] {inShape[0], inShape[1] * inShape[2] * inShape[3]};

        INDArray ret = input.reshape('c', outShape);

        Pair<INDArray, MaskState> p = feedForwardMaskArray(activations.getMask(0), activations.getMaskState(0), miniBatchSize);
        return ActivationsFactory.getInstance().create(ret, p.getFirst(), p.getSecond() );
    }

    @Override
    public Gradients backprop(Gradients gradients, int miniBatchSize) {
        if(gradients.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess activation gradients: Activation gradients must have " +
                    "exactly 1 array. Got: " + gradients.size());
        }
        INDArray epsilons = gradients.get(0);
        //Epsilons from layer above should be 2d, with shape [miniBatchSize, depthOut*outH*outW]
        if (epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons))
            epsilons = epsilons.dup('c');

        if (epsilons.rank() == 4)
            return gradients; //Should never happen

        if (epsilons.columns() != inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows "
                            + inputHeight + " x columns " + inputWidth + " x depth " + numChannels + " but was instead "
                            + Arrays.toString(epsilons.shape()));

        INDArray ret = epsilons.reshape('c', epsilons.size(0), numChannels, inputHeight, inputWidth);

        return GradientsFactory.getInstance().create(ret, gradients.getParameterGradients());
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

    @Override
    public InputType[] getOutputType(InputType... inputType) {
        if (inputType == null || inputType.length != 1 || inputType[0].getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got "
                    + (inputType == null ? null : Arrays.toString(inputType)));
        }

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType[0];
        int outSize = c.getDepth() * c.getHeight() * c.getWidth();
        return new InputType[]{InputType.feedForward(outSize)};
    }



    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Pass-through, unmodified (assuming here that it's a 1d mask array - one value per example)
        return new Pair<>(maskArray, currentMaskState);
    }
}
