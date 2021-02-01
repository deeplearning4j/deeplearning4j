/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.preprocessor;

import lombok.Data;
import lombok.val;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

@Data
public class CnnToFeedForwardPreProcessor implements InputPreProcessor {
    protected long inputHeight;
    protected long inputWidth;
    protected long numChannels;
    protected CNN2DFormat format = CNN2DFormat.NCHW;    //Default for legacy JSON deserialization

    /**
     * @param inputHeight the columns
     * @param inputWidth the rows
     * @param numChannels the channels
     */

    @JsonCreator
    public CnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                    @JsonProperty("inputWidth") long inputWidth, @JsonProperty("numChannels") long numChannels,
                        @JsonProperty("format") CNN2DFormat format) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        if(format != null)
            this.format = format;
    }

    public CnnToFeedForwardPreProcessor(long inputHeight, long inputWidth) {
        this(inputHeight, inputWidth, 1, CNN2DFormat.NCHW);
    }

    public CnnToFeedForwardPreProcessor(long inputHeight, long inputWidth, long numChannels) {
        this(inputHeight, inputWidth, numChannels, CNN2DFormat.NCHW);
    }

    public CnnToFeedForwardPreProcessor() {}

    @Override
    // return 2 dimensions
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() == 2)
            return input; //Should usually never happen

        int chDim = 1;
        int hDim = 2;
        int wDim = 3;
        if(format == CNN2DFormat.NHWC){
            chDim = 3;
            hDim = 1;
            wDim = 2;
        }

        if(inputHeight == 0 && inputWidth == 0 && numChannels == 0){
            this.inputHeight = input.size(hDim);
            this.inputWidth = input.size(wDim);
            this.numChannels = input.size(chDim);
        }

        if(input.size(chDim) != numChannels || input.size(hDim) != inputHeight || input.size(wDim) != inputWidth){
            throw new IllegalStateException("Invalid input, does not match configuration: expected " +
                    (format == CNN2DFormat.NCHW ? "[minibatch, numChannels=" + numChannels + ", inputHeight=" + inputHeight + ", inputWidth=" + inputWidth + "] " :
                            "[minibatch, inputHeight=" + inputHeight + ", inputWidth=" + inputWidth + ", numChannels=" + numChannels + "]") +
                            " but got input array of shape " + Arrays.toString(input.shape()));
        }

        //Check input: nchw format
        if(input.size(chDim) != numChannels || input.size(hDim) != inputHeight ||
                input.size(wDim) != inputWidth){
            throw new IllegalStateException("Invalid input array: expected shape [minibatch, channels, height, width] = "
                    + "[minibatch, " + numChannels + ", " + inputHeight + ", " + inputWidth + "] - got "
                    + Arrays.toString(input.shape()));
        }

        //Assume input is standard rank 4 activations out of CNN layer
        //First: we require input to be in c order. But c order (as declared in array order) isn't enough; also need strides to be correct
        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');

        //Note that to match Tensorflow/Keras, we do a simple "c order reshape" for both NCHW and NHWC

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

        INDArray ret;
        if(format == CNN2DFormat.NCHW){
            ret = epsilons.reshape('c', epsilons.size(0), numChannels, inputHeight, inputWidth);
        } else {
            ret = epsilons.reshape('c', epsilons.size(0), inputHeight, inputWidth, numChannels);
        }

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
        //h=2,w=1,c=5 pre processor: 0,0,NCHW (broken)
        //h=2,w=2,c=3, cnn=2,2,3, NCHW
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
