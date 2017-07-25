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

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * Adds the ability to reshape and flatten the tensor in the computation graph.
 *
 * @author Justin Long (crockpotveggies)
 */
@Data
public class ReshapeVertex extends GraphVertex {

    public ReshapeVertex(@JsonProperty("newShape") int... newShape) {
        this.newShape = newShape;
    }

    protected int[] newShape;

    @Override
    public ReshapeVertex clone() {
        return new ReshapeVertex(newShape);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ReshapeVertex))
            return false;
        return Arrays.equals(((ReshapeVertex) o).newShape, newShape);
    }

    @Override
    public int hashCode() {
        return newShape.hashCode();
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
        return new org.deeplearning4j.nn.graph.vertex.impl.ReshapeVertex(graph, name, idx, newShape);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length == 1)
            return vertexInputs[0];
        InputType first = vertexInputs[0];
        if (first.getType() != InputType.Type.CNN) {
            //FF, RNN or flat CNN data inputs
            for (int i = 1; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != first.getType()) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: Reshape vertex cannot process activations of different types:"
                                                    + " first type = " + first.getType() + ", input type " + (i + 1)
                                                    + " = " + vertexInputs[i].getType());
                }
            }
        } else {
            //CNN inputs... also check that the depth, width and heights match:
            InputType.InputTypeConvolutional firstConv = (InputType.InputTypeConvolutional) first;
            int fd = firstConv.getDepth();
            int fw = firstConv.getWidth();
            int fh = firstConv.getHeight();

            for (int i = 1; i < vertexInputs.length; i++) {
                if (vertexInputs[i].getType() != InputType.Type.CNN) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: Reshape vertex cannot process activations of different types:"
                                                    + " first type = " + InputType.Type.CNN + ", input type " + (i + 1)
                                                    + " = " + vertexInputs[i].getType());
                }

                InputType.InputTypeConvolutional otherConv = (InputType.InputTypeConvolutional) vertexInputs[i];

                int od = otherConv.getDepth();
                int ow = otherConv.getWidth();
                int oh = otherConv.getHeight();

                if (fd != od || fw != ow || fh != oh) {
                    throw new InvalidInputTypeException(
                                    "Invalid input: Reshape vertex cannot process CNN activations of different sizes:"
                                                    + "first [depth,width,height] = [" + fd + "," + fw + "," + fh
                                                    + "], input " + i + " = [" + od + "," + ow + "," + oh + "]");
                }
            }
        }
        return first; //Same output shape/size as
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //Assume it's a reshape-with-copy op. In this case: memory use is accounted for in activations
        InputType outputType = getOutputType(-1, inputTypes);
        return new LayerMemoryReport.Builder(null, ReshapeVertex.class, inputTypes[0], outputType).standardMemory(0, 0) //No params
                        .workingMemory(0, 0, 0, 0).cacheMemory(0, 0) //No caching
                        .build();
    }
}
