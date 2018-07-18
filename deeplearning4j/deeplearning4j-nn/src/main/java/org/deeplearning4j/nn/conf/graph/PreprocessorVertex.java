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

package org.deeplearning4j.nn.conf.graph;


import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
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

    /**
     * @param preProcessor The input preprocessor
     */
    public PreprocessorVertex(InputPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public GraphVertex clone() {
        return new PreprocessorVertex(preProcessor.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof PreprocessorVertex))
            return false;
        return ((PreprocessorVertex) o).preProcessor.equals(preProcessor);
    }

    @Override
    public int hashCode() {
        return preProcessor.hashCode();
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 1;
    }

    @Override
    public int maxVertexInputs() {
        return 1;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                    INDArray paramsView, boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex(graph, name, idx, preProcessor);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1)
            throw new InvalidInputTypeException("Invalid input: Preprocessor vertex expects " + "exactly one input");

        return preProcessor.getOutputType(vertexInputs[0]);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //TODO: eventually account for preprocessor memory use

        InputType outputType = getOutputType(-1, inputTypes);
        return new LayerMemoryReport.Builder(null, PreprocessorVertex.class, inputTypes[0], outputType)
                        .standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
