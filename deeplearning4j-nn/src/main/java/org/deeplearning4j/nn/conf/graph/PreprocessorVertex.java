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
import lombok.NonNull;
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

import java.util.Collection;

/** PreprocessorVertex is a simple adaptor class that allows a {@link InputPreProcessor} to be used in a ComputationGraph
 * GraphVertex, without it being associated with a layer.
 * @author Alex Black
 */
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = false)
public class PreprocessorVertex extends BaseGraphVertex {

    /**
     * @param preProcessor The input preprocessor
     */
    public PreprocessorVertex(@NonNull InputPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public PreprocessorVertex clone() {
        return new PreprocessorVertex(preProcessor.clone());
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
    public Layer instantiate(Collection<IterationListener> iterationListeners,
                             String name, int idx, int numInputs, INDArray layerParamsView,
                             boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex(name, idx, numInputs, preProcessor);
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1)
            throw new InvalidInputTypeException("Invalid input: Preprocessor vertex expects " + "exactly one input");

        return preProcessor.getOutputType(vertexInputs[0]);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        //TODO: eventually account for preprocessor memory use

        InputType outputType = getOutputType(-1, inputTypes)[0];
        return new LayerMemoryReport.Builder(null, PreprocessorVertex.class, inputTypes[0], outputType)
                        .standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
