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

package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

public class UnstackVertex extends BaseGraphVertex {
    private long from;
    private int stackSize;
    private long forwardShape[];
    private long step;

    public UnstackVertex(ComputationGraph graph, String name, int vertexIndex, int from, int stackSize, DataType dataType) {
        this(graph, name, vertexIndex, null, null, from, stackSize, dataType);
    }

    public UnstackVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, int from, int stackSize, DataType dataType) {
        super(graph, name, vertexIndex, inputVertices, outputVertices, dataType);
        this.from = from;
        this.stackSize = stackSize;
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: input not set");

        // once we know the inputs, save the shape and interval size for doBackward
        this.forwardShape = Arrays.copyOf(inputs[0].shape(), inputs[0].rank());

        this.step = inputs[0].size(0) / stackSize;
        long start = from * step;
        long end = (from + 1) * step;

        INDArray ret;
        switch (inputs[0].rank()) { //TODO remove the dups here if/when possible (gradient checks must pass)
            case 2:
                ret = inputs[0].get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
                break;
            case 3:
                ret = inputs[0].get(NDArrayIndex.interval(start, end), NDArrayIndex.all(), NDArrayIndex.all());
                break;
            case 4:
                ret = inputs[0].get(NDArrayIndex.interval(start, end), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all());
                break;
            default:
                throw new UnsupportedOperationException(
                                "Cannot get subset for activations of rank " + inputs[0].rank());
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: error not set");

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, inputs[0].dataType(), forwardShape);
        long start = from * step;
        long end = (from + 1) * step;

        switch (forwardShape.length) {
            case 2:
                out.put(new INDArrayIndex[] {NDArrayIndex.interval(start, end), NDArrayIndex.all()}, epsilon);
                break;
            case 3:
                out.put(new INDArrayIndex[] {NDArrayIndex.interval(start, end), NDArrayIndex.all(), NDArrayIndex.all()},
                                epsilon);
                break;
            case 4:
                out.put(new INDArrayIndex[] {NDArrayIndex.interval(start, end), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all()}, epsilon);
                break;
            default:
                throw new RuntimeException("Invalid activation rank"); //Should never happen
        }
        return new Pair<>(null, new INDArray[] {out});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        if (maskArrays == null || maskArrays.length == 0) {
            return new Pair<>(null, currentMaskState);
        }

        boolean allNull = true;
        for (int i = 0; i < maskArrays.length; i++) {
            if (maskArrays[i] != null) {
                allNull = false;
                break;
            }
        }
        if (allNull) {
            return new Pair<>(null, currentMaskState);
        }

        //Mask arrays are either 1d (column vector) or 2d...
        long start = from * minibatchSize;
        long end = (from + 1) * minibatchSize;
        INDArray outMask = maskArrays[0].get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
        return new Pair<>(outMask, currentMaskState);
    }

    @Override
    public String toString() {
        return "UnstackVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",fromIdx=" + from
                        + ",forwardShape=" + forwardShape + ")";
    }
}
