/*
 *
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4j.nn.conf.graph;


import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/** PreprocessorVertex is a simple adaptor class that allows a {@link InputPreProcessor} to be used in a ComputationGraph
 * GraphVertex, without it being associated with a layer.
 * @author Alex Black
 */
@NoArgsConstructor
@Data
public class PreprocessorVertex extends GraphVertex {

    private InputPreProcessor preProcessor;
    private InputType outputType;

    public PreprocessorVertex(InputPreProcessor preProcessor) {
        this(preProcessor, null);
    }

    /**
     * @param preProcessor The input preprocessor
     * @param outputType Override for the type of output used in {@link #getOutputType(InputType...)}. This may be necessary
     *                   for the automatic addition of other processors in the network, given a custom/non-standard InputPreProcessor
     */
    public PreprocessorVertex(InputPreProcessor preProcessor, InputType outputType) {
        this.preProcessor = preProcessor;
        this.outputType = outputType;
    }

    @Override
    public GraphVertex clone() {
        return new PreprocessorVertex(preProcessor.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof PreprocessorVertex)) return false;
        return ((PreprocessorVertex) o).preProcessor.equals(preProcessor);
    }

    @Override
    public int hashCode() {
        return preProcessor.hashCode();
    }

    @Override
    public int numParams(boolean backprop){
        return 0;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView) {
        return new org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex(graph, name, idx, preProcessor);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) throw new InvalidInputTypeException("Invalid input: Preprocessor vertex expects "
                + "exactly one input");
        if (outputType != null) return outputType;   //Allows user to override for custom preprocessors

        //Otherwise, try to infer:
        switch (vertexInputs[0].getType()) {
            case FF:
                if (preProcessor instanceof FeedForwardToCnnPreProcessor) {
                    FeedForwardToCnnPreProcessor ffcnn = (FeedForwardToCnnPreProcessor) preProcessor;
                    return InputType.convolutional(ffcnn.getNumChannels(), ffcnn.getInputWidth(), ffcnn.getInputHeight());
                } else if (preProcessor instanceof FeedForwardToRnnPreProcessor) {
                    return InputType.recurrent(((InputType.InputTypeFeedForward)vertexInputs[0]).getSize());
                } else {
                    //Assume preprocessor doesn't change the type of activations
                    return InputType.feedForward(((InputType.InputTypeFeedForward) vertexInputs[0]).getSize());
                }
            case RNN:
                if (preProcessor instanceof RnnToCnnPreProcessor) {
                    RnnToCnnPreProcessor ffcnn = (RnnToCnnPreProcessor) preProcessor;
                    return InputType.convolutional(ffcnn.getNumChannels(), ffcnn.getInputWidth(), ffcnn.getInputHeight());
                } else if (preProcessor instanceof RnnToFeedForwardPreProcessor) {
                    return InputType.feedForward(((InputType.InputTypeRecurrent) vertexInputs[0]).getSize());
                } else {
                    //Assume preprocessor doesn't change the type of activations
                    return InputType.recurrent(((InputType.InputTypeRecurrent)vertexInputs[0]).getSize());
                }
            case CNN:
                if (preProcessor instanceof CnnToFeedForwardPreProcessor) {
                    CnnToFeedForwardPreProcessor p = (CnnToFeedForwardPreProcessor)preProcessor;
                    int outSize = p.getInputHeight()*p.getInputWidth()*p.getNumChannels();
                    return InputType.feedForward(outSize);
                } else if (preProcessor instanceof CnnToRnnPreProcessor) {
                    CnnToRnnPreProcessor p = (CnnToRnnPreProcessor)preProcessor;
                    int outSize = p.getInputHeight()*p.getInputWidth()*p.getNumChannels();
                    return InputType.recurrent(outSize);
                } else {
                    //Assume preprocessor doesn't change the type of activations
                    return vertexInputs[0];
                }
            default:
                throw new RuntimeException("Unknown InputType: " + vertexInputs[0]);
        }

    }
}
