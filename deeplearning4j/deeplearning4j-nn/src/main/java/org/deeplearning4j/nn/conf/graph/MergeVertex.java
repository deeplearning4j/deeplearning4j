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


import lombok.val;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/** A MergeVertex is used to combine the activations of two or more layers/GraphVertex by means of concatenation/merging.<br>
 * Exactly how this is done depends on the type of input.<br>
 * For 2d (feed forward layer) inputs: {@code MergeVertex([numExamples,layerSize1],[numExamples,layerSize2]) -> [numExamples,layerSize1 + layerSize2]}<br>
 * For 3d (time series) inputs: {@code MergeVertex([numExamples,layerSize1,timeSeriesLength],[numExamples,layerSize2,timeSeriesLength])
 *      -> [numExamples,layerSize1 + layerSize2,timeSeriesLength]}<br>
 * For 4d (convolutional) inputs: {@code MergeVertex([numExamples,depth1,width,height],[numExamples,depth2,width,height])
 *      -> [numExamples,depth1 + depth2,width,height]}<br>
 * @author Alex Black
 */
public class MergeVertex extends GraphVertex {

    @Override
    public MergeVertex clone() {
        return new MergeVertex();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof MergeVertex;
    }

    @Override
    public int hashCode() {
        return 433682566;
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 2;
    }

    @Override
    public int maxVertexInputs() {
        return Integer.MAX_VALUE;
    }

    @Override
    public String toString() {
        return "MergeVertex()";
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                    INDArray paramsView, boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.MergeVertex(graph, name, idx);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length == 1)
            return vertexInputs[0];
        InputType first = vertexInputs[0];
        if (first.getType() == InputType.Type.CNNFlat) {
            //TODO
            //Merging flattened CNN format data could be messy?
            throw new InvalidInputTypeException(
                            "Invalid input: MergeVertex cannot currently merge CNN data in flattened format. Got: "
                                            + vertexInputs);
        } else if (first.getType() != InputType.Type.CNN) {
            //FF or RNN data inputs
            int size = 0;
            InputType.Type type = null;
            for (int i = 0; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != first.getType()) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: MergeVertex cannot merge activations of different types:"
                                                    + " first type = " + first.getType() + ", input type " + (i + 1)
                                                    + " = " + vertexInputs[i].getType());
                }

                long thisSize;
                switch (vertexInputs[i].getType()) {
                    case FF:
                        thisSize = ((InputType.InputTypeFeedForward) vertexInputs[i]).getSize();
                        type = InputType.Type.FF;
                        break;
                    case RNN:
                        thisSize = ((InputType.InputTypeRecurrent) vertexInputs[i]).getSize();
                        type = InputType.Type.RNN;
                        break;
                    default:
                        throw new IllegalStateException("Unknown input type: " + vertexInputs[i]); //Should never happen
                }
                if (thisSize <= 0) {//Size is not defined
                    size = -1;
                } else {
                    size += thisSize;
                }
            }

            if (size > 0) {
                //Size is specified
                if (type == InputType.Type.FF) {
                    return InputType.feedForward(size);
                } else {
                    val tsLength = ((InputType.InputTypeRecurrent) vertexInputs[0]).getTimeSeriesLength();
                    return InputType.recurrent(size, tsLength);
                }
            } else {
                //size is unknown
                if (type == InputType.Type.FF) {
                    return InputType.feedForward(-1);
                } else {
                    val tsLength = ((InputType.InputTypeRecurrent) vertexInputs[0]).getTimeSeriesLength();
                    return InputType.recurrent(-1, tsLength);
                }
            }
        } else {
            //CNN inputs... also check that the channels, width and heights match:
            InputType.InputTypeConvolutional firstConv = (InputType.InputTypeConvolutional) first;

            // FIXME: int cast
            val fd = (int) firstConv.getChannels();
            val fw = (int) firstConv.getWidth();
            val fh = (int) firstConv.getHeight();

            int depthSum = fd;

            for (int i = 1; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != InputType.Type.CNN) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: MergeVertex cannot process activations of different types:"
                                                    + " first type = " + InputType.Type.CNN + ", input type " + (i + 1)
                                                    + " = " + vertexInputs[i].getType());
                }

                InputType.InputTypeConvolutional otherConv = (InputType.InputTypeConvolutional) vertexInputs[i];

                // FIXME: int cast
                val od = (int) otherConv.getChannels();
                val ow = (int) otherConv.getWidth();
                val oh = (int) otherConv.getHeight();

                if (fw != ow || fh != oh) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: MergeVertex cannot merge CNN activations of different width/heights:"
                                                    + "first [channels,width,height] = [" + fd + "," + fw + "," + fh
                                                    + "], input " + i + " = [" + od + "," + ow + "," + oh + "]");
                }

                depthSum += od;
            }

            return InputType.convolutional(fh, fw, depthSum);
        }
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        InputType outputType = getOutputType(-1, inputTypes);

        //TODO multiple input types
        return new LayerMemoryReport.Builder(null, MergeVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0) //No working memory in addition to activations/epsilons
                        .cacheMemory(0, 0) //No caching
                        .build();
    }
}
