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

package org.deeplearning4j.nn.conf.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;

/**
 * LayerVertex is a GraphVertex with a neural network Layer (and, optionally an {@link InputPreProcessor}) in it
 *
 * @author Alex Black
 */
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = false)
public class LayerVertex extends BaseGraphVertex {

    private NeuralNetConfiguration layerConf;
    //Set outputVertex to true when Layer is an OutputLayer, OR For use in specialized situations like reinforcement learning
    // For RL situations, this Layer insn't an OutputLayer, but is the last layer in a graph, that gets its error/epsilon
    // passed in externally
    private boolean outputVertex;


    public LayerVertex(NeuralNetConfiguration layerConf) {
        this.layerConf = layerConf;
    }

    @Override
    public LayerVertex clone() {
        return new LayerVertex(layerConf.clone());
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
    public Layer instantiate(NeuralNetConfiguration conf,
                             Collection<IterationListener> iterationListeners,
                             String name, int layerIndex, int numInputs, INDArray layerParamsView,
                             boolean initializeParams) {

        org.deeplearning4j.nn.api.Layer layer =
                        layerConf.getLayer().instantiate(layerConf, iterationListeners, name, layerIndex, numInputs,
                                layerParamsView, initializeParams);

        return layer;
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        return layerConf.getLayer().getOutputType(layerIndex, vertexInputs);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        //TODO preprocessor memory
        return layerConf.getLayer().getMemoryReport(inputTypes[0]);
    }
}
