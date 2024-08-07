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


import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class StackVertex extends GraphVertex {

    public StackVertex() {}

    @Override
    public StackVertex clone() {
        return new StackVertex();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof StackVertex;
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
        return Integer.MAX_VALUE;
    }

    @Override
    public int hashCode() {
        return 433682566;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams, DataType networkDatatype) {
        return new org.deeplearning4j.nn.graph.vertex.impl.StackVertex(graph, name, idx, networkDatatype);
    }

    @Override
    public String toString() {
        return "StackVertex()";
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length == 1)
            return vertexInputs[0];
        InputType first = vertexInputs[0];

        //Check that types are all the same...
        for( int i = 1; i < vertexInputs.length; i++) {
            Preconditions.checkState(vertexInputs[i].getType() == first.getType(), "Different input types found:" +
                    " input types must be the same. First type: %s, type %s: %s", first, i, vertexInputs[i]);

            //Check that types are equal:
            Preconditions.checkState(first.equals(vertexInputs[i]), "Input types must be equal: %s and %s", first,
                    vertexInputs[i]);
        }

        //Stacking on dimension 0 -> same output type as input type
        return first;
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory, just output activations
        InputType outputType = getOutputType(-1, inputTypes);

        return new LayerMemoryReport.Builder(null, StackVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
