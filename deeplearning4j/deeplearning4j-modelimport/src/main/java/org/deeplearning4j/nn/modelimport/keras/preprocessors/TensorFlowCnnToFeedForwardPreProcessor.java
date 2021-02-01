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

package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * @deprecated Exists only for backward compatibility of older pretrained models. Should not be used.
 * Use {@link CnnToFeedForwardPreProcessor} for all new models instead.
 */
@Slf4j @Deprecated
public class TensorFlowCnnToFeedForwardPreProcessor extends CnnToFeedForwardPreProcessor {

    @JsonCreator @Deprecated
    public TensorFlowCnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                                                  @JsonProperty("inputWidth") long inputWidth,
                                                  @JsonProperty("numChannels") long numChannels) {
        super(inputHeight, inputWidth, numChannels);
    }

    @Deprecated
    public TensorFlowCnnToFeedForwardPreProcessor(long inputHeight, long inputWidth) {
        super(inputHeight, inputWidth);
    }

    @Deprecated
    public TensorFlowCnnToFeedForwardPreProcessor() {
        super();
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() == 2)
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input); //Should usually never happen
        /* DL4J convolutional input:       # channels, # rows, # cols
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */
        INDArray permuted = workspaceMgr.dup(ArrayType.ACTIVATIONS, input.permute(0, 2, 3, 1), 'c'); //To: [n, h, w, c]

        val inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        val outShape = new long[]{inShape[0], inShape[1] * inShape[2] * inShape[3]};

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, permuted.reshape('c', outShape));
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (epsilons.ordering() != 'c' || !Shape.hasDefaultStridesForShape(epsilons))
            epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');

        INDArray epsilonsReshaped = epsilons.reshape('c', epsilons.size(0), inputHeight, inputWidth, numChannels);

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonsReshaped.permute(0, 3, 1, 2));    //To [n, c, h, w]
    }

    @Override
    public TensorFlowCnnToFeedForwardPreProcessor clone() {
        return (TensorFlowCnnToFeedForwardPreProcessor) super.clone();
    }
}