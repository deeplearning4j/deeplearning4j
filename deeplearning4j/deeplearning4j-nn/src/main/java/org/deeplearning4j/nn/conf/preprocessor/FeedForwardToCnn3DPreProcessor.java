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
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

import static org.nd4j.linalg.api.shape.Shape.hasDefaultStridesForShape;

/**
 * A preprocessor to allow 3D CNN and standard feed-forward network layers to be used together.<br>
 * For example, DenseLayer -> Convolution3D<br>
 * This does two things:<br>
 * (a) Reshapes activations out of FeedFoward layer (which is 2D with shape
 * [numExamples, inputDepth*inputHeight*inputWidth*numChannels]) into 5d activations (with shape
 * [numExamples, numChannels, inputDepth, inputHeight, inputWidth]) suitable to feed into CNN layers.<br>
 * (b) Reshapes 5d epsilons from 3D CNN layer, with shape
 * [numExamples, numChannels, inputDepth, inputHeight, inputWidth]) into 2d epsilons (with shape
 * [numExamples, inputDepth*inputHeight*inputWidth*numChannels]) for use in feed forward layer
 *
 * @author MaxPumperla
 * @see CnnToFeedForwardPreProcessor for opposite case (i.e., CNN3D -> DenseLayer etc)
 */
@Data
@EqualsAndHashCode(exclude = {"shape"})
public class FeedForwardToCnn3DPreProcessor implements InputPreProcessor {
    private int inputDepth;
    private int inputHeight;
    private int inputWidth;
    private int numChannels;
    private boolean isNCDHW = true;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private long[] shape;

    /**
     * @param inputDepth  input channels
     * @param inputHeight input height
     * @param inputWidth  input width
     * @param numChannels input channels
     * @param isNCDHW     boolean to indicate data format, i.e. channels first (NCDHW) vs. channels last (NDHWC)
     */
    @JsonCreator
    public FeedForwardToCnn3DPreProcessor(@JsonProperty("inputDepth") int inputDepth,
                                          @JsonProperty("inputHeight") int inputHeight,
                                          @JsonProperty("inputWidth") int inputWidth,
                                          @JsonProperty("numChannels") int numChannels,
                                          @JsonProperty("isNCDHW") boolean isNCDHW) {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.isNCDHW = isNCDHW;
    }

    public FeedForwardToCnn3DPreProcessor(int inputDepth, int inputWidth, int inputHeight) {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        this.shape = input.shape();

        if (shape.length == 5)
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input);

        if (!hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');

        if (input.columns() != inputDepth * inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to channels "
                    + inputDepth + " times height " + inputWidth + "times width " + inputWidth
                    + " times channels " + numChannels
                    + " but was instead " + Arrays.toString(input.shape()));

        INDArray ret;
        if (isNCDHW)
            ret = input.reshape('c', input.size(0), numChannels, inputDepth, inputHeight, inputWidth);
        else
            ret = input.reshape('c', input.size(0), inputDepth, inputHeight, inputWidth, numChannels);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (!hasDefaultStridesForShape(epsilons))
            epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');

        if (shape == null || ArrayUtil.prod(shape) != epsilons.length()) {
            INDArray ret = epsilons.reshape('c', epsilons.size(0),inputDepth * inputHeight * inputWidth * numChannels);
            return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, ret);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilons.reshape('c', shape));
    }


    @Override
    public FeedForwardToCnn3DPreProcessor clone() {
        try {
            FeedForwardToCnn3DPreProcessor clone = (FeedForwardToCnn3DPreProcessor) super.clone();
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
                int expSize = inputDepth * inputHeight * inputWidth * numChannels;
                if (c.getSize() != expSize) {
                    throw new IllegalStateException("Invalid input: expected FeedForward input of size " + expSize
                            + " = (d=" + numChannels + " * w=" + inputWidth + " * h=" + inputHeight + "), got "
                            + inputType);
                }
                return InputType.convolutional3D(inputDepth, inputHeight, inputWidth, numChannels);
            case CNN:
                InputType.InputTypeConvolutional c2 = (InputType.InputTypeConvolutional) inputType;

                if (c2.getChannels() != numChannels || c2.getHeight() != inputHeight || c2.getWidth() != inputWidth) {
                    throw new IllegalStateException("Invalid input: Got CNN input type with (c,w,h)=(" + c2.getChannels()
                            + "," + c2.getWidth() + "," + c2.getHeight() + ") but expected (" + numChannels
                            + "," + inputHeight + "," + inputWidth + ")");
                }
                return InputType.convolutional3D(1, c2.getHeight(), c2.getWidth(), c2.getChannels());
            case CNN3D:
                InputType.InputTypeConvolutional3D c3 = (InputType.InputTypeConvolutional3D) inputType;

                if (c3.getChannels() != numChannels || c3.getDepth() != inputDepth ||
                        c3.getHeight() != inputHeight || c3.getWidth() != inputWidth) {
                    throw new IllegalStateException("Invalid input: Got CNN input type with (c, d,w,h)=("
                            + c3.getChannels() + "," + c3.getDepth() + "," + c3.getWidth() + "," + c3.getHeight()
                            + ") but expected (" + numChannels + "," + inputDepth + ","
                            + inputHeight + "," + inputWidth + ")");
                }
                return c3;
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
