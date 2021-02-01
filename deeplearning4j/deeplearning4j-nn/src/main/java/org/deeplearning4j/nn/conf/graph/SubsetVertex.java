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
import lombok.val;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

@Data
public class SubsetVertex extends GraphVertex {

    private int from;
    private int to;

    /**
     * @param from The first column index, inclusive
     * @param to   The last column index, inclusive
     */
    public SubsetVertex(@JsonProperty("from") int from, @JsonProperty("to") int to) {
        this.from = from;
        this.to = to;
    }

    @Override
    public SubsetVertex clone() {
        return new SubsetVertex(from, to);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof SubsetVertex))
            return false;
        SubsetVertex s = (SubsetVertex) o;
        return s.from == from && s.to == to;
    }

    @Override
    public int hashCode() {
        return new Integer(from).hashCode() ^ new Integer(to).hashCode();
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
        return new org.deeplearning4j.nn.graph.vertex.impl.SubsetVertex(graph, name, idx, from, to, networkDatatype);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException(
                            "SubsetVertex expects single input type. Received: " + Arrays.toString(vertexInputs));
        }

        switch (vertexInputs[0].getType()) {
            case FF:
                return InputType.feedForward(to - from + 1);
            case RNN:
                return InputType.recurrent(to - from + 1);
            case CNN:
                InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) vertexInputs[0];
                val depth = conv.getChannels();
                if (to >= depth) {
                    throw new InvalidInputTypeException("Invalid range: Cannot select channels subset [" + from + "," + to
                                    + "] inclusive from CNN activations with " + " [channels,width,height] = [" + depth
                                    + "," + conv.getWidth() + "," + conv.getHeight() + "]");
                }
                return InputType.convolutional(conv.getHeight(), conv.getWidth(), from - to + 1);
            case CNNFlat:
                //TODO work out how to do this - could be difficult...
                throw new UnsupportedOperationException(
                                "Subsetting data in flattened convolutional format not yet supported");
            default:
                throw new RuntimeException("Unknown input type: " + vertexInputs[0]);
        }
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //Get op without dup - no additional memory use
        InputType outputType = getOutputType(-1, inputTypes);
        return new LayerMemoryReport.Builder(null, SubsetVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
