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

import lombok.Data;
import lombok.val;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

import static org.nd4j.linalg.api.shape.Shape.hasDefaultStridesForShape;

/**
 * A preprocessor to allow CNN and standard feed-forward network layers to be used together.<br>
 * For example, CNN3D -> Denselayer <br>
 * This does two things:<br>
 * (b) Reshapes 5d activations out of CNN layer, with shape
 * [numExamples, numChannels, inputDepth, inputHeight, inputWidth]) into 2d activations (with shape
 * [numExamples, inputDepth*inputHeight*inputWidth*numChannels]) for use in feed forward layer
 * (a) Reshapes epsilons (weights*deltas) out of FeedFoward layer (which is 2D or 3D with shape
 * [numExamples, inputDepth* inputHeight*inputWidth*numChannels]) into 5d epsilons (with shape
 * [numExamples, numChannels, inputDepth, inputHeight, inputWidth]) suitable to feed into CNN layers.<br>
 * Note: numChannels is equivalent to featureMaps referenced in different literature
 *
 * @author Max Pumperla
 * @see FeedForwardToCnn3DPreProcessor for opposite case (i.e., DenseLayer -> CNN3D)
 */
@Data
public class Cnn3DToFeedForwardPreProcessor implements InputPreProcessor {
    protected long inputDepth;
    protected long inputHeight;
    protected long inputWidth;
    protected long numChannels;
    protected boolean isNCDHW = true; // channels first ordering by default

    /**
     * @param inputDepth  input channels
     * @param inputHeight input height
     * @param inputWidth  input width
     * @param numChannels input channels
     * @param isNCDHW     boolean to indicate data format, i.e. channels first (NCDHW) vs. channels last (NDHWC)
     */
    @JsonCreator
    public Cnn3DToFeedForwardPreProcessor(@JsonProperty("inputDepth") long inputDepth,
                                          @JsonProperty("inputHeight") long inputHeight,
                                          @JsonProperty("inputWidth") long inputWidth,
                                          @JsonProperty("numChannels") long numChannels,
                                          @JsonProperty("isNCDHW") boolean isNCDHW) {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.isNCDHW = isNCDHW;
    }

    public Cnn3DToFeedForwardPreProcessor(int inputDepth, int inputHeight, int inputWidth) {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    public Cnn3DToFeedForwardPreProcessor() {
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() == 2)
            return input; // Pass-through feed-forward input

        // We expect either NCDHW or NDHWC format
        if ((isNCDHW && input.size(1) != numChannels) || (!isNCDHW && input.size(4) != numChannels)) {
            throw new IllegalStateException("Invalid input array: expected shape in format "
                    + "[minibatch, channels, channels, height, width] or "
                    + "[minibatch, channels, height, width, channels]"
                    + "for numChannels: " + numChannels + ", inputDepth " + inputDepth + ", inputHeight " + inputHeight
                    + " and inputWidth " + inputWidth + ", but got "
                    + Arrays.toString(input.shape()));
        }

        if (!hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');

        val inShape = input.shape();
        val outShape = new long[]{inShape[0], inShape[1] * inShape[2] * inShape[3] * inShape[4]};

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input.reshape('c', outShape));
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Epsilons are 2d, with shape [miniBatchSize, outChannels*outD*outH*outW]

        if (!hasDefaultStridesForShape(epsilons))
            epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');

        if (epsilons.rank() == 5)
            return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilons); //Should never happen

        if (epsilons.columns() != inputDepth * inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output to have depth: "
                    + inputDepth + ", height: " + inputHeight + ", width: " + inputWidth + " and channels: "
                    + numChannels + ", i.e. [" + epsilons.rows() + ", "
                    + inputDepth * inputHeight * inputWidth * numChannels + "] but was instead "
                    + Arrays.toString(epsilons.shape()));

        INDArray ret;
        if (isNCDHW)
            ret = epsilons.reshape('c', epsilons.size(0), numChannels, inputDepth, inputHeight, inputWidth);
        else
            ret = epsilons.reshape('c', epsilons.size(0), inputDepth, inputHeight, inputWidth, numChannels);

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, ret); //Move to specified workspace if required

    }

    @Override
    public Cnn3DToFeedForwardPreProcessor clone() {
        try {
            Cnn3DToFeedForwardPreProcessor clone = (Cnn3DToFeedForwardPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN3D, got " + inputType);
        }

        InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
        val outSize = c.getChannels() * c.getDepth() * c.getHeight() * c.getWidth();
        return InputType.feedForward(outSize);
    }


    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        //Pass-through, unmodified (assuming here that it's a 1d mask array - one value per example)
        return new Pair<>(maskArray, currentMaskState);
    }
}
