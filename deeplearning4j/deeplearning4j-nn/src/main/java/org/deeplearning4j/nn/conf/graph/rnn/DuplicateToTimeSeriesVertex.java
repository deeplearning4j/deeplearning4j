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

package org.deeplearning4j.nn.conf.graph.rnn;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * DuplicateToTimeSeriesVertex is a vertex that goes from 2d activations to a 3d time series activations, by means of
 * duplication. That is, given a 2d input with shape [numExamples,nIn] duplicate each row to give output of
 * [numExamples,nIn,timeSeriesLength], where the activations are the same for all time steps.<br>
 * This method is used for example in sequence to sequence models.<br>
 * <b>Note</b>: The length of the output time series (number of time steps) is determined by means of referencing one of the
 * inputs in the ComputationGraph. That is: Because the length of the time series may differ at runtime, we generally want the number
 * of time steps to match some other input; here, we are specifying the length of the output time series to be the same as
 * one of the input time series<br>
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class DuplicateToTimeSeriesVertex extends GraphVertex {

    private String inputName;

    /**
     * @param inputName Name of the input in the ComputationGraph network to use, to determine how long the output time
     *                  series should be. This input should (a) exist, and (b) be a time series input
     */
    public DuplicateToTimeSeriesVertex(@JsonProperty("inputName") String inputName) {
        this.inputName = inputName;
    }

    @Override
    public GraphVertex clone() {
        return new DuplicateToTimeSeriesVertex(inputName);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DuplicateToTimeSeriesVertex))
            return false;
        DuplicateToTimeSeriesVertex d = (DuplicateToTimeSeriesVertex) o;
        if (inputName == null && d.inputName != null || inputName != null && d.inputName == null)
            return false;
        return inputName == null || inputName.equals(d.inputName);
    }

    @Override
    public int hashCode() {
        return 534806565 ^ (inputName != null ? inputName.hashCode() : 0);
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
        return new org.deeplearning4j.nn.graph.vertex.impl.rnn.DuplicateToTimeSeriesVertex(graph, name, idx, inputName);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1)
            throw new InvalidInputTypeException("Invalid input type: cannot duplicate more than 1 input");

        int tsLength = 1; //TODO work this out properly

        if (vertexInputs[0].getType() == InputType.Type.FF) {
            return InputType.recurrent(((InputType.InputTypeFeedForward) vertexInputs[0]).getSize(), tsLength);
        } else if (vertexInputs[0].getType() == InputType.Type.CNNFlat) {
            return InputType.recurrent(((InputType.InputTypeConvolutionalFlat) vertexInputs[0]).getFlattenedSize(),
                            tsLength);
        } else {
            throw new InvalidInputTypeException(
                            "Invalid input type: cannot duplicate to time series non feed forward (or CNN flat) input (got: "
                                            + vertexInputs[0] + ")");
        }


    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, DuplicateToTimeSeriesVertex.class, inputTypes[0],
                        getOutputType(-1, inputTypes)).standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0)
                                        .build();
    }
}
