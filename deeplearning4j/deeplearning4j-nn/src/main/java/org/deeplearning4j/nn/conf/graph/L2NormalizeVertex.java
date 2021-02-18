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

package org.deeplearning4j.nn.conf.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.val;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
@EqualsAndHashCode(callSuper = false)
public class L2NormalizeVertex extends GraphVertex {
    public static final double DEFAULT_EPS = 1e-8;

    protected int[] dimension;
    protected double eps;

    public L2NormalizeVertex() {
        this(null, DEFAULT_EPS);
    }

    public L2NormalizeVertex(@JsonProperty("dimension") int[] dimension, @JsonProperty("eps") double eps) {
        this.dimension = dimension;
        this.eps = eps;
    }



    @Override
    public L2NormalizeVertex clone() {
        return new L2NormalizeVertex(dimension, eps);
    }

    @Override
    public long numParams(boolean backprop) {
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
                                                                      INDArray paramsView, boolean initializeParams, DataType networkDatatype) {

        return new org.deeplearning4j.nn.graph.vertex.impl.L2NormalizeVertex(graph, name, idx, dimension, eps, networkDatatype);
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
        InputType outputType = getOutputType(-1, inputTypes);
        //norm2 value (inference working mem): 1 per example during forward pass

        //Training working mem: 2 per example + 2x input size + 1 per example (in addition to epsilons)
        val trainModePerEx = 3 + 2 * inputTypes[0].arrayElementsPerExample();

        return new LayerMemoryReport.Builder(null, L2NormalizeVertex.class, inputTypes[0], outputType)
                        .standardMemory(0, 0) //No params
                        .workingMemory(0, 1, 0, trainModePerEx).cacheMemory(0, 0) //No caching
                        .build();
    }
}
