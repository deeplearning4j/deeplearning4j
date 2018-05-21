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

package org.deeplearning4j.nn.conf;


import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.serde.legacyformat.LegacyPreprocessorDeserializerHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Input pre processor used
 * for pre processing input before passing it
 * to the neural network.
 *
 * @author Adam Gibson
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyPreprocessorDeserializerHelper.class)
public interface InputPreProcessor extends Serializable, Cloneable {

    /**
     * Pre preProcess input/activations for a multi layer network
     * @param input the input to pre preProcess
     * @param miniBatchSize Minibatch size
     * @param workspaceMgr Workspace manager
     * @return the processed input. Note that the returned array should be placed in the
     *         {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATIONS} workspace via the workspace manager
     */
    INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr);

    /**Reverse the preProcess during backprop. Process Gradient/epsilons before
     * passing them to the layer below.
     * @param output which is a pair of the gradient and epsilon
     * @param miniBatchSize Minibatch size
     * @param workspaceMgr Workspace manager
     * @return the reverse of the pre preProcess step (if any). Note that the returned array should be
     *         placed in {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATION_GRAD} workspace via the
     *         workspace manager
     */
    INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr);

    InputPreProcessor clone();

    /**
     * For a given type of input to this preprocessor, what is the type of the output?
     *
     * @param inputType    Type of input for the preprocessor
     * @return             Type of input after applying the preprocessor
     */
    InputType getOutputType(InputType inputType);


    Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize);
}
