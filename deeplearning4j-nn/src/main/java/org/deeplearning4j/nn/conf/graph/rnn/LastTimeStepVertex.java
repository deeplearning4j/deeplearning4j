/*-
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

package org.deeplearning4j.nn.conf.graph.rnn;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collection;

/** LastTimeStepVertex is used in the context of recurrent neural network activations, to go from 3d (time series)
 * activations to 2d activations, by extracting out the last time step of activations for each example.<br>
 * This can be used for example in sequence to sequence architectures, and potentially for sequence classification.
 * <b>NOTE</b>: Because RNNs may have masking arrays (to allow for examples/time series of different lengths in the same
 * minibatch), it is necessary to provide the same of the network input that has the corresponding mask array. If this
 * input does not have a mask array, the last time step of the input will be used for all examples; otherwise, the time
 * step of the last non-zero entry in the mask array (for each example separately) will be used.
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class LastTimeStepVertex extends GraphVertex {

    @Override
    public GraphVertex clone() {
        return new LastTimeStepVertex();
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minInputs() {
        return 1;
    }

    @Override
    public int maxInputs() {
        return 1;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.impl.rnn.LastTimeStepVertex instantiate(NeuralNetConfiguration conf,
                                                                                      Collection<IterationListener> iterationListeners,
                                                                                      String name, int idx, int numInputs, INDArray layerParamsView,
                                                                                      boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.rnn.LastTimeStepVertex(name, idx, numInputs);
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1)
            throw new InvalidInputTypeException("Invalid input type: cannot get last time step of more than 1 input");
        if (vertexInputs[0].getType() != InputType.Type.RNN) {
            throw new InvalidInputTypeException(
                            "Invalid input type: cannot get subset of non RNN input (got: " + vertexInputs[0] + ")");
        }

        return new InputType[]{InputType.feedForward(((InputType.InputTypeRecurrent) vertexInputs[0]).getSize())};
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        //No additional working memory (beyond activations/epsilons)
        return new LayerMemoryReport.Builder(null, LastTimeStepVertex.class, inputTypes[0],
                        getOutputType(-1, inputTypes)[0])
                .standardMemory(0, 0)
                .workingMemory(0, 0, 0, 0)
                .cacheMemory(0, 0)
                                        .build();
    }

    @Override
    public String toString() {
        return "LastTimeStepVertex()";
    }
}
