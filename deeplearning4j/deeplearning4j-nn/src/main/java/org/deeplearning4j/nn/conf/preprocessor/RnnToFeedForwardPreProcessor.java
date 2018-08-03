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
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

/**
 * A preprocessor to allow RNN and feed-forward network layers to be used together.<br>
 * For example, GravesLSTM -> OutputLayer or GravesLSTM -> DenseLayer<br>
 * This does two things:<br>
 * (a) Reshapes activations out of RNN layer (which is 3D with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) into 2d activations (with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) suitable for use in feed-forward layers.<br>
 * (b) Reshapes 2d epsilons (weights*deltas from feed forward layer, with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) into 3d epsilons (with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) for use in RNN layer
 *
 * @author Alex Black
 * @see FeedForwardToRnnPreProcessor for opposite case (i.e., DenseLayer -> GravesLSTM etc)
 */
@Data
@Slf4j
public class RnnToFeedForwardPreProcessor implements InputPreProcessor {

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Need to reshape RNN activations (3d) activations to 2d (for input into feed forward layer)
        if (input.rank() != 3)
            throw new IllegalArgumentException(
                            "Invalid input: expect NDArray with rank 3 (i.e., activations for RNN layer)");

        if (input.ordering() != 'f' || !Shape.hasDefaultStridesForShape(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        val shape = input.shape();
        INDArray ret;
        if (shape[0] == 1) {
            ret = input.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        } else if (shape[2] == 1) {
            ret = input.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            INDArray permuted = input.permute(0, 2, 1); //Permute, so we get correct order after reshaping
            ret = permuted.reshape('f', shape[0] * shape[2], shape[1]);
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (output == null)
            return null; //In a few cases: output may be null, and this is valid. Like time series data -> embedding layer
        //Need to reshape FeedForward layer epsilons (2d) to 3d (for use in RNN layer backprop calculations)
        if (output.rank() != 2)
            throw new IllegalArgumentException(
                            "Invalid input: expect NDArray with rank 2 (i.e., epsilons from feed forward layer)");
        if (output.ordering() != 'f' || !Shape.hasDefaultStridesForShape(output))
            output = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output, 'f');

        val shape = output.shape();
        INDArray reshaped = output.reshape('f', miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, reshaped.permute(0, 2, 1));
    }

    @Override
    public RnnToFeedForwardPreProcessor clone() {
        try {
            RnnToFeedForwardPreProcessor clone = (RnnToFeedForwardPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input: expected input of type RNN, got " + inputType);
        }

        InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType;
        return InputType.feedForward(rnn.getSize());
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Assume mask array is 2d for time series (1 value per time step)
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else if (maskArray.rank() == 2) {
            //Need to reshape mask array from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1]
            return new Pair<>(TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray, LayerWorkspaceMgr.noWorkspaces(), ArrayType.INPUT),  //TODO
                    currentMaskState);
        } else {
            throw new IllegalArgumentException("Received mask array of rank " + maskArray.rank()
                            + "; expected rank 2 mask array. Mask array shape: " + Arrays.toString(maskArray.shape()));
        }
    }
}
