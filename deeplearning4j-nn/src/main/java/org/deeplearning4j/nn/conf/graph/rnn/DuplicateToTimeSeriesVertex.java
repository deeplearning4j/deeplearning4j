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

package org.deeplearning4j.nn.conf.graph.rnn;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Collection;

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

    public DuplicateToTimeSeriesVertex() {
    }

    @Override
    public GraphVertex clone() {
        return new DuplicateToTimeSeriesVertex();
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minInputs() {
        return 2;
    }

    @Override
    public int maxInputs() {
        return 2;
    }

    @Override
    public Layer instantiate(Collection<IterationListener> iterationListeners,
                             String name, int idx, int numInputs, INDArray layerParamsView,
                             boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.rnn.DuplicateToTimeSeriesVertex( name, idx, numInputs);
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 2)
            throw new InvalidInputTypeException("Invalid input type: expected 2 inputs (rank 2 to duplicate + " +
                    "rank 3 to determine output size). Got: " + Arrays.toString(vertexInputs));

        InputType it2d = null;
        InputType it3d = null;

        if (vertexInputs[0].getType() == InputType.Type.FF || vertexInputs[0].getType() == InputType.Type.CNNFlat) {
            it2d = vertexInputs[0];
        } else if(vertexInputs[0].getType() == InputType.Type.RNN ){
            it3d = vertexInputs[0];
        }

        if (vertexInputs[1].getType() == InputType.Type.FF || vertexInputs[1].getType() == InputType.Type.CNNFlat) {
            it2d = vertexInputs[1];
        } else if(vertexInputs[1].getType() == InputType.Type.RNN ){
            it3d = vertexInputs[1];
        }

        if(it2d == null || it3d == null){
            throw new InvalidInputTypeException("Invalid input type: expected 2 inputs (rank 2 to duplicate + " +
                    "rank 3 to determine output size). Got: " + Arrays.toString(vertexInputs));
        }

        int outSize = it2d.arrayElementsPerExample();
        int tsLength = ((InputType.InputTypeRecurrent)it3d).getTimeSeriesLength();

        return new InputType[]{InputType.recurrent(outSize, tsLength)};
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, DuplicateToTimeSeriesVertex.class, inputTypes[0],
                        getOutputType(-1, inputTypes)[0])
                .standardMemory(0, 0)
                .workingMemory(0, 0, 0, 0)
                .cacheMemory(0, 0).build();
    }
}
