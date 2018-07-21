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

import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.graph.rnn.ReverseTimeSeriesVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.serde.legacyformat.LegacyGraphVertexDeserializerHelper;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * A GraphVertex is a vertex in the computation graph. It may contain Layer, or define some arbitrary forward/backward pass
 * behaviour based on the inputs
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyGraphVertexDeserializerHelper.class)
public abstract class GraphVertex implements Cloneable, Serializable {

    @Override
    public abstract GraphVertex clone();

    @Override
    public abstract boolean equals(Object o);

    @Override
    public abstract int hashCode();

    public abstract int numParams(boolean backprop);

    /**
     * @return The Smallest valid number of inputs to this vertex
     */
    public abstract int minVertexInputs();

    /**
     * @return The largest valid number of inputs to this vertex
     */
    public abstract int maxVertexInputs();

    /**
     * Create a {@link org.deeplearning4j.nn.graph.vertex.GraphVertex} instance, for the given computation graph,
     * given the configuration instance.
     *
     * @param graph            The computation graph that this GraphVertex is to be part of
     * @param name             The name of the GraphVertex object
     * @param idx              The index of the GraphVertex
     * @param paramsView       A view of the full parameters array
     * @param initializeParams If true: initialize the parameters. If false: make no change to the values in the paramsView array
     * @return The implementation GraphVertex object (i.e., implementation, no the configuration)
     */
    public abstract org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name,
                    int idx, INDArray paramsView, boolean initializeParams);

    /**
     * Determine the type of output for this GraphVertex, given the specified inputs. Given that a GraphVertex may do arbitrary
     * processing or modifications of the inputs, the output types can be quite different to the input type(s).<br>
     * This is generally used to determine when to add preprocessors, as well as the input sizes etc for layers
     *
     * @param layerIndex The index of the layer (if appropriate/necessary).
     * @param vertexInputs The inputs to this vertex
     * @return The type of output for this vertex
     * @throws InvalidInputTypeException If the input type is invalid for this type of GraphVertex
     */
    public abstract InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException;

    /**
     * This is a report of the estimated memory consumption for the given vertex
     *
     * @param inputTypes Input types to the vertex. Memory consumption is often a function of the input type
     * @return Memory report for the vertex
     */
    public abstract MemoryReport getMemoryReport(InputType... inputTypes);

}
