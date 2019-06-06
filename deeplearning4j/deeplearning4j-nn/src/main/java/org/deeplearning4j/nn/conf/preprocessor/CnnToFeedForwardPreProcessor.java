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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
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
 * Note: numChannels is equivalent to channels or featureMaps referenced in different literature
 * @author Adam Gibson
 * @see FeedForwardToCnnPreProcessor for opposite case (i.e., DenseLayer -> CNNetc)
 */
@Data
public class CnnToFeedForwardPreProcessor implements InputPreProcessor {
    protected long inputHeight;
    protected long inputWidth;
    protected long numChannels;

    /**
     * @param inputHeight the columns
     * @param inputWidth the rows
     * @param numChannels the channels
     */

    @JsonCreator
    public CnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                    @JsonProperty("inputWidth") long inputWidth, @JsonProperty("numChannels") long numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    public CnnToFeedForwardPreProcessor(long inputHeight, long inputWidth) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = 1;
    }

    public CnnToFeedForwardPreProcessor() {}

    @Override
    // return 2 dimensions
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() == 2)
            return input; //Should usually never happen
        if(input.size(1) != numChannels || input.size(2) != inputHeight || input.size(3) != inputWidth){
            throw new IllegalStateException("Invalid input, does not match configuration: expected [minibatch, numChannels="
                    + numChannels + ", inputHeight=" + inputHeight + ", inputWidth=" + inputWidth + "] but got input array of" +
                    "shape " + Arrays.toString(input.shape()));
        }

        //Check input: nchw format
        if(input.size(1) != numChannels || input.size(2) != inputHeight ||
                input.size(3) != inputWidth){
            throw new IllegalStateException("Invalid input array: expected shape [minibatch, channels, height, width] = "
                    + "[minibatch, " + numChannels + ", " + inputHeight + ", " + inputWidth + "] - got "
                    + Arrays.toString(input.shape()));
        }

        //Assume input is standard rank 4 activations out of CNN layer
        //First: we require input to be in c order. But c order (as declared in array order) isn't enough; also need strides to be correct
        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');

        val inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        val outShape = new long[]{inShape[0], inShape[1] * inShape[2] * inShape[3]};

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input.reshape('c', outShape));    //Should be zero copy reshape
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Epsilons from layer above should be 2d, with shape [miniBatchSize, depthOut*outH*outW]
        if (epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons))
            epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');

        if (epsilons.rank() == 4)
            return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilons); //Should never happen

        if (epsilons.columns() != inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows "
                            + inputHeight + " x columns " + inputWidth + " x channels " + numChannels + " but was instead "
                            + Arrays.toString(epsilons.shape()));

        INDArray ret = epsilons.reshape('c', epsilons.size(0), numChannels, inputHeight, inputWidth);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, ret); //Move if required to specified workspace
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
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got " + inputType);
        }

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        val outSize = c.getChannels() * c.getHeight() * c.getWidth();
        return InputType.feedForward(outSize);
    }


    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        if(maskArray == null || maskArray.rank() == 2)
            return new Pair<>(maskArray, currentMaskState);

        if (maskArray.rank() != 4 || maskArray.size(2) != 1 || maskArray.size(3) != 1) {
            throw new UnsupportedOperationException(
                    "Expected rank 4 mask array for 2D CNN layer activations. Got rank " + maskArray.rank() + " mask array (shape " +
                            Arrays.toString(maskArray.shape()) + ")  - when used in conjunction with input data of shape" +
                            " [batch,channels,h,w] 4d masks passing through CnnToFeedForwardPreProcessor should have shape" +
                            " [batchSize,1,1,1]");
        }

        return new Pair<>(maskArray.reshape(maskArray.ordering(), maskArray.size(0), maskArray.size(1)), currentMaskState);
    }
}
