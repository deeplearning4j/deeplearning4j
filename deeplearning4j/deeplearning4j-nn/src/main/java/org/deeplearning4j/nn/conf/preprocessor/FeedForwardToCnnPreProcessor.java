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

package org.deeplearning4j.nn.conf.preprocessor;

import lombok.*;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * A preprocessor to allow CNN and standard feed-forward network layers to be used together.<br>
 * For example, DenseLayer -> CNN<br>
 * This does two things:<br>
 * (a) Reshapes activations out of FeedFoward layer (which is 2D or 3D with shape
 * [numExamples, inputHeight*inputWidth*numChannels]) into 4d activations (with shape
 * [numExamples, numChannels, inputHeight, inputWidth]) suitable to feed into CNN layers.<br>
 * (b) Reshapes 4d epsilons (weights*deltas) from CNN layer, with shape
 * [numExamples, numChannels, inputHeight, inputWidth]) into 2d epsilons (with shape
 * [numExamples, inputHeight*inputWidth*numChannels]) for use in feed forward layer
 * Note: numChannels is equivalent to channels or featureMaps referenced in different literature
 *
 * @author Adam Gibson
 * @see Cnn3DToFeedForwardPreProcessor for opposite case (i.e., CNN -> DenseLayer etc)
 */
@Data
@EqualsAndHashCode(exclude = {"shape"})
public class FeedForwardToCnnPreProcessor implements InputPreProcessor {
    private long inputHeight;
    private long inputWidth;
    private long numChannels;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private long[] shape;

    /**
     * Reshape to a channels x rows x columns tensor
     *
     * @param inputHeight the columns
     * @param inputWidth  the rows
     * @param numChannels the channels
     */
    @JsonCreator
    public FeedForwardToCnnPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                    @JsonProperty("inputWidth") long inputWidth, @JsonProperty("numChannels") long numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    public FeedForwardToCnnPreProcessor(long inputWidth, long inputHeight) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        this.shape = input.shape();
        if (input.rank() == 4)
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input);

        if (input.columns() != inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows "
                    + inputHeight + " x columns " + inputWidth + " x channels " + numChannels
                    + " but was instead " + Arrays.toString(input.shape()));

        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS,
                input.reshape('c', input.size(0), numChannels, inputHeight, inputWidth));
    }

    @Override
    // return 4 dimensions
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (epsilons.ordering() != 'c' || !Shape.hasDefaultStridesForShape(epsilons))
            epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');

        if (shape == null || ArrayUtil.prod(shape) != epsilons.length()) {
            if (epsilons.rank() == 2)
                return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilons); //should never happen

            return epsilons.reshape('c', epsilons.size(0), numChannels, inputHeight, inputWidth);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilons.reshape('c', shape));
    }


    @Override
    public FeedForwardToCnnPreProcessor clone() {
        try {
            FeedForwardToCnnPreProcessor clone = (FeedForwardToCnnPreProcessor) super.clone();
            if (clone.shape != null)
                clone.shape = clone.shape.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {

        switch (inputType.getType()) {
            case FF:
                InputType.InputTypeFeedForward c = (InputType.InputTypeFeedForward) inputType;
                val expSize = inputHeight * inputWidth * numChannels;
                if (c.getSize() != expSize) {
                    throw new IllegalStateException("Invalid input: expected FeedForward input of size " + expSize
                                    + " = (d=" + numChannels + " * w=" + inputWidth + " * h=" + inputHeight + "), got "
                                    + inputType);
                }
                return InputType.convolutional(inputHeight, inputWidth, numChannels);
            case CNN:
                InputType.InputTypeConvolutional c2 = (InputType.InputTypeConvolutional) inputType;

                if (c2.getChannels() != numChannels || c2.getHeight() != inputHeight || c2.getWidth() != inputWidth) {
                    throw new IllegalStateException("Invalid input: Got CNN input type with (d,w,h)=(" + c2.getChannels()
                                    + "," + c2.getWidth() + "," + c2.getHeight() + ") but expected (" + numChannels
                                    + "," + inputHeight + "," + inputWidth + ")");
                }
                return c2;
            case CNNFlat:
                InputType.InputTypeConvolutionalFlat c3 = (InputType.InputTypeConvolutionalFlat) inputType;
                if (c3.getDepth() != numChannels || c3.getHeight() != inputHeight || c3.getWidth() != inputWidth) {
                    throw new IllegalStateException("Invalid input: Got CNN input type with (d,w,h)=(" + c3.getDepth()
                                    + "," + c3.getWidth() + "," + c3.getHeight() + ") but expected (" + numChannels
                                    + "," + inputHeight + "," + inputWidth + ")");
                }
                return c3.getUnflattenedType();
            default:
                throw new IllegalStateException("Invalid input type: got " + inputType);
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Pass-through, unmodified (assuming here that it's a 1d mask array - one value per example)
        return new Pair<>(maskArray, currentMaskState);
    }

}
