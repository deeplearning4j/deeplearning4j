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
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A ShiftVertex is used to shift the activations of a single layer. It is addition of a scalar value.<br>
 * One could use it to add a bias or as part of some other calculation.
 * For example, Highway Layers need them in two places. One, it's often
 * useful to have the gate weights have a large negative bias. (Of course
 * for this, we could just initialize the biases that way.)
 * But, _also_ it needs to do this:
 * {@code (1-sigmoid(weight * input + bias)) (*) input + sigmoid(weight * input + bias) (*) activation(w2 * input + bias) ((*) is hadamard product)}
 * <br>
 * So, here, we could have:<br>
 * 1. a DenseLayer that does the sigmoid<br>
 * 2. a ScaleVertex(-1) and<br>
 * 3. a ShiftVertex(1)<br>
 * to accomplish that.<br>
 *
 * @author Binesh Bannerjee (binesh_binesh@hotmail.com, @bnsh on gitter)
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = false)
public class ShiftVertex extends GraphVertex {

    public ShiftVertex(@JsonProperty("shiftFactor") double shiftFactor) {
        this.shiftFactor = shiftFactor;
    }

    protected double shiftFactor = 0.0; // Shift by zero if it's not specified.

    @Override
    public ShiftVertex clone() {
        return new ShiftVertex(shiftFactor);
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

        return new org.deeplearning4j.nn.graph.vertex.impl.ShiftVertex(graph, name, idx, shiftFactor);
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
        return new LayerMemoryReport.Builder(null, ShiftVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
