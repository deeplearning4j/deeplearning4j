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


import lombok.Getter;
import lombok.val;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * UnstackVertex allows for unstacking of inputs so that they may be forwarded through
 * a network. This is useful for cases such as Triplet Embedding, where embeddings can
 * be separated and run through subsequent layers.
 *
 * Works similarly to SubsetVertex, except on dimension 0 of the input.
 *
 * @author Justin Long (crockpotveggies)
 */
@Getter
public class UnstackVertex extends GraphVertex {
    protected int from;
    protected int stackSize;

    /**
     * @param from The first column index of the stacked inputs.
     * @param stackSize The total number of stacked inputs. An interval is automatically
     *                  calculated according to the size of the first dimension.
     */
    public UnstackVertex(@JsonProperty("from") int from, @JsonProperty("stackSize") int stackSize) {
        this.from = from;
        this.stackSize = stackSize;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                    INDArray paramsView, boolean initializeParams) {
        return new org.deeplearning4j.nn.graph.vertex.impl.UnstackVertex(graph, name, idx, null, null, from, stackSize);
    }

    @Override
    public UnstackVertex clone() {
        return new UnstackVertex(from, stackSize);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof UnstackVertex))
            return false;
        return ((UnstackVertex) o).from == from && ((UnstackVertex) o).stackSize == stackSize;
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
    public int hashCode() {
        return 433682566;
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
                            "Invalid input: UnstackVertex cannot currently merge CNN data in flattened format. Got: "
                                            + vertexInputs);
        } else if (first.getType() != InputType.Type.CNN) {
            //FF or RNN data inputs
            int size = 0;
            InputType.Type type = null;
            for (int i = 0; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != first.getType()) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: UnstackVertex cannot merge activations of different types:"
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
                if (type == InputType.Type.FF)
                    return InputType.feedForward(size);
                else
                    return InputType.recurrent(size);
            } else {
                //size is unknown
                if (type == InputType.Type.FF)
                    return InputType.feedForward(-1);
                else
                    return InputType.recurrent(-1);
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
                                    "Invalid input: UnstackVertex cannot process activations of different types:"
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
                                    "Invalid input: UnstackVertex cannot merge CNN activations of different width/heights:"
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
        //Get op with dup - accounted for in activations size (no working memory)
        //Do one dup on the forward pass (output activations). Accounted for in output activations.
        InputType outputType = getOutputType(-1, inputTypes);
        return new LayerMemoryReport.Builder(null, UnstackVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
