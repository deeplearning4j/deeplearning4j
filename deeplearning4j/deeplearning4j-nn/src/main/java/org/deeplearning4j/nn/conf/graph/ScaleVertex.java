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
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A ScaleVertex is used to scale the size of activations of a single layer<br>
 * For example, ResNet activations can be scaled in repeating blocks to keep variance
 * under control.
 *
 * @author Justin Long (@crockpotveggies)
 */
@Data
public class ScaleVertex extends GraphVertex {

    public ScaleVertex(@JsonProperty("scaleFactor") double scaleFactor) {
        this.scaleFactor = scaleFactor;
    }

    protected double scaleFactor;

    @Override
    public ScaleVertex clone() {
        return new ScaleVertex(scaleFactor);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ScaleVertex))
            return false;
        return ((ScaleVertex) o).scaleFactor == scaleFactor;
    }

    @Override
    public int hashCode() {
        return 123073088;
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

        return new org.deeplearning4j.nn.graph.vertex.impl.ScaleVertex(graph, name, idx, scaleFactor);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length == 1)
            return vertexInputs[0];
        InputType first = vertexInputs[0];

        return first; //Same output shape/size as
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //Do one dup on the forward pass (output activations). Accounted for in output activations.
        InputType outputType = getOutputType(-1, inputTypes);
        return new LayerMemoryReport.Builder(null, ScaleVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
