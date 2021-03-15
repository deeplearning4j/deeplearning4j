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

import lombok.*;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

@Data
@EqualsAndHashCode(exclude = {"product"})
public class CnnToRnnPreProcessor implements InputPreProcessor {
    private long inputHeight;
    private long inputWidth;
    private long numChannels;
    private RNNFormat rnnDataFormat = RNNFormat.NCW;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private long product;

    @JsonCreator
    public CnnToRnnPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                                @JsonProperty("inputWidth") long inputWidth,
                                @JsonProperty("numChannels") long numChannels,
                                @JsonProperty("rnnDataFormat") RNNFormat rnnDataFormat) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.product = inputHeight * inputWidth * numChannels;
        this.rnnDataFormat = rnnDataFormat;
    }

    public CnnToRnnPreProcessor(long inputHeight,
                                long inputWidth,
                                long numChannels){
        this(inputHeight, inputWidth, numChannels, RNNFormat.NCW);
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() != 4)
            throw new IllegalArgumentException(
                            "Invalid input: expect CNN activations with rank 4 (received input with shape "
                                            + Arrays.toString(input.shape()) + ")");
        if(input.size(1) != numChannels || input.size(2) != inputHeight || input.size(3) != inputWidth){
            throw new IllegalStateException("Invalid input, does not match configuration: expected [minibatch, numChannels="
                    + numChannels + ", inputHeight=" + inputHeight + ", inputWidth=" + inputWidth + "] but got input array of" +
                    "shape " + Arrays.toString(input.shape()));
        }
        //Input: 4d activations (CNN)
        //Output: 3d activations (RNN)

        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = input.dup('c');

        val shape = input.shape(); //[timeSeriesLength*miniBatchSize, numChannels, inputHeight, inputWidth]

        //First: reshape 4d to 2d, as per CnnToFeedForwardPreProcessor
        INDArray twod = input.reshape('c', input.size(0), ArrayUtil.prod(input.shape()) / input.size(0));
        //Second: reshape 2d to 3d, as per FeedForwardToRnnPreProcessor
        INDArray reshaped = workspaceMgr.dup(ArrayType.ACTIVATIONS, twod, 'f');
        reshaped = reshaped.reshape('f', miniBatchSize, shape[0] / miniBatchSize, product);
        if (rnnDataFormat == RNNFormat.NCW) {
            return reshaped.permute(0, 2, 1);
        }
        return reshaped;
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (output.ordering() == 'c' || !Shape.hasDefaultStridesForShape(output))
            output = output.dup('f');
        if (rnnDataFormat == RNNFormat.NWC){
            output = output.permute(0, 2, 1);
        }
        val shape = output.shape();
        INDArray output2d;
        if (shape[0] == 1) {
            //Edge case: miniBatchSize = 1
            output2d = output.tensorAlongDimension(0, 1, 2).permutei(1, 0);
        } else if (shape[2] == 1) {
            //Edge case: timeSeriesLength = 1
            output2d = output.tensorAlongDimension(0, 1, 0);
        } else {
            //As per FeedForwardToRnnPreprocessor
            INDArray permuted3d = output.permute(0, 2, 1);
            output2d = permuted3d.reshape('f', shape[0] * shape[2], shape[1]);
        }

        if (shape[1] != product)
            throw new IllegalArgumentException("Invalid input: expected output size(1)=" + shape[1]
                            + " must be equal to " + inputHeight + " x columns " + inputWidth + " x channels "
                            + numChannels + " = " + product + ", received: " + shape[1]);
        INDArray ret = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output2d, 'c');
        return ret.reshape('c', output2d.size(0), numChannels, inputHeight, inputWidth);
    }

    @Override
    public CnnToRnnPreProcessor clone() {
        return new CnnToRnnPreProcessor(inputHeight, inputWidth, numChannels, rnnDataFormat);
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got " + inputType);
        }

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        val outSize = c.getChannels() * c.getHeight() * c.getWidth();
        return InputType.recurrent(outSize, rnnDataFormat);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Assume mask array is 4d - a mask array that has been reshaped from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1, 1, 1]
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else {
            //Need to reshape mask array from [minibatch*timeSeriesLength, 1, 1, 1] to [minibatch,timeSeriesLength]
            return new Pair<>(TimeSeriesUtils.reshapeCnnMaskToTimeSeriesMask(maskArray, minibatchSize),currentMaskState);
        }
    }
}
