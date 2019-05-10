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

package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Preprocessor to flatten input of RNN type
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
public class KerasFlattenRnnPreprocessor extends BaseInputPreProcessor {

    private long tsLength;
    private long depth;

    public KerasFlattenRnnPreprocessor(@JsonProperty("depth") long depth, @JsonProperty("tsLength") long tsLength) {
        super();
        this.tsLength = Math.abs(tsLength);
        this.depth = depth;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        INDArray output = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');
        return output.reshape(input.size(0), depth * tsLength);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c').reshape(miniBatchSize, depth, tsLength);
    }

    @Override
    public KerasFlattenRnnPreprocessor clone() {
        return (KerasFlattenRnnPreprocessor) super.clone();
    }

    @Override
    public InputType getOutputType(InputType inputType) throws InvalidInputTypeException {

        return InputType.feedForward(depth * tsLength);

    }
}
