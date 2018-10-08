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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * FrozenVertex is used for the purposes of transfer learning.<br>
 * A frozen vertex wraps another DL4J GraphVertex within it.
 * During backprop, the FrozenVertex is skipped, and any parameters are not be updated.
 * Usually users will typically not create FrozenVertex instances directly - they are usually used in the process of performing
 * transfer learning
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class FrozenVertex extends GraphVertex {

    private GraphVertex underlying;

    public FrozenVertex(@JsonProperty("underlying") GraphVertex underlying){
        this.underlying = underlying;
    }

    @Override
    public GraphVertex clone() {
        return new FrozenVertex(underlying.clone());
    }

    @Override
    public int numParams(boolean backprop) {
        return underlying.numParams(backprop);
    }

    @Override
    public int minVertexInputs() {
        return underlying.minVertexInputs();
    }

    @Override
    public int maxVertexInputs() {
        return underlying.maxVertexInputs();
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        org.deeplearning4j.nn.graph.vertex.GraphVertex u = underlying.instantiate(graph, name, idx, paramsView, initializeParams);
        return new org.deeplearning4j.nn.graph.vertex.impl.FrozenVertex(u);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        return underlying.getOutputType(layerIndex, vertexInputs);
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return underlying.getMemoryReport(inputTypes);
    }
}
