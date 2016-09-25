/*
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


import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Input pre processor used
 * for pre processing input before passing it
 * to the neural network.
 *
 * @author Adam Gibson
 */
@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = CnnToFeedForwardPreProcessor.class, name = "cnnToFeedForward"),
        @JsonSubTypes.Type(value = CnnToRnnPreProcessor.class, name = "cnnToRnn"),
        @JsonSubTypes.Type(value = ComposableInputPreProcessor.class, name = "composableInput"),
        @JsonSubTypes.Type(value = FeedForwardToCnnPreProcessor.class, name = "feedForwardToCnn"),
        @JsonSubTypes.Type(value = FeedForwardToRnnPreProcessor.class, name = "feedForwardToRnn"),
        @JsonSubTypes.Type(value = RnnToFeedForwardPreProcessor.class, name = "rnnToFeedForward"),
        @JsonSubTypes.Type(value = RnnToCnnPreProcessor.class, name = "rnnToCnn"),
        @JsonSubTypes.Type(value = BinomialSamplingPreProcessor.class, name = "binomialSampling"),
        @JsonSubTypes.Type(value = ReshapePreProcessor.class, name = "reshape"),
        @JsonSubTypes.Type(value = UnitVarianceProcessor.class, name = "unitVariance"),
        @JsonSubTypes.Type(value = ZeroMeanAndUnitVariancePreProcessor.class, name = "zeroMeanAndUnitVariance"),
        @JsonSubTypes.Type(value = ZeroMeanPrePreProcessor.class, name = "zeroMean"),
})
public interface InputPreProcessor extends Serializable, Cloneable {


    /**
     * Pre preProcess input/activations for a multi layer network
     * @param input the input to pre preProcess
     * @param miniBatchSize
     * @return the processed input
     */
    INDArray preProcess(INDArray input, int miniBatchSize);

    /**Reverse the preProcess during backprop. Process Gradient/epsilons before
     * passing them to the layer below.
     * @param output which is a pair of the gradient and epsilon
     * @param miniBatchSize
     * @return the reverse of the pre preProcess step (if any)
     */
    INDArray backprop(INDArray output, int miniBatchSize);

    InputPreProcessor clone();

    /**
     * For a given type of input to this preprocessor, what is the type of the output?
     *
     * @param inputType    Type of input for the preprocessor
     * @return             Type of input after applying the preprocessor
     */
    InputType getOutputType(InputType inputType);
}
