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
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Composable input pre processor
 * @author Adam Gibson
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class ComposableInputPreProcessor extends BaseInputPreProcessor {
    private InputPreProcessor[] inputPreProcessors;

    @JsonCreator
    public ComposableInputPreProcessor(@JsonProperty("inputPreProcessors") InputPreProcessor... inputPreProcessors) {
        this.inputPreProcessors = inputPreProcessors;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        for (InputPreProcessor preProcessor : inputPreProcessors)
            input = preProcessor.preProcess(input, miniBatchSize, workspaceMgr);
        return input;
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Apply input preprocessors in opposite order for backprop (compared to forward pass)
        //For example, CNNtoFF + FFtoRNN, need to do backprop in order of FFtoRNN + CNNtoFF
        for (int i = inputPreProcessors.length - 1; i >= 0; i--) {
            output = inputPreProcessors[i].backprop(output, miniBatchSize, workspaceMgr);
        }
        return output;
    }

    @Override
    public ComposableInputPreProcessor clone() {
        ComposableInputPreProcessor clone = (ComposableInputPreProcessor) super.clone();
        if (clone.inputPreProcessors != null) {
            InputPreProcessor[] processors = new InputPreProcessor[clone.inputPreProcessors.length];
            for (int i = 0; i < clone.inputPreProcessors.length; i++) {
                processors[i] = clone.inputPreProcessors[i].clone();
            }
            clone.inputPreProcessors = processors;
        }
        return clone;
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        for (InputPreProcessor p : inputPreProcessors) {
            inputType = p.getOutputType(inputType);
        }
        return inputType;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        for (InputPreProcessor preproc : inputPreProcessors) {
            Pair<INDArray, MaskState> p = preproc.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);
            maskArray = p.getFirst();
            currentMaskState = p.getSecond();
        }
        return new Pair<>(maskArray, currentMaskState);
    }
}
