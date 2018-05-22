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

package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

/**
 * UnstackVertex allows for unstacking of inputs so that they may be forwarded through
 * a network. This is useful for cases such as Triplet Embedding, where embeddings can
 * be separated and run through subsequent layers.
 *
 * Works similarly to SubsetVertex, except on dimension 0 of the input. stackSize is
 * explicitly defined by the user to properly calculate an step.
 *
 * @author Justin Long (crockpotveggies)
 */
public class UnstackVertex extends BaseGraphVertex {
    private int from;
    private int stackSize;
    private long forwardShape[];
    private int step;

    public UnstackVertex(ComputationGraph graph, String name, int vertexIndex, int from, int stackSize) {
        this(graph, name, vertexIndex, null, null, from, stackSize);
    }

    public UnstackVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, int from, int stackSize) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
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

        // FIXME: int cast
        this.step = (int) inputs[0].size(0) / stackSize;
        int start = from * step;
        int end = (from + 1) * step;

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

        return workspaceMgr.dup(ArrayType.ACTIVATIONS, ret);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: error not set");

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, forwardShape);
        int start = from * step;
        int end = (from + 1) * step;

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
        int start = from * minibatchSize;
        int end = (from + 1) * minibatchSize;
        INDArray outMask = maskArrays[0].get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
        return new Pair<>(outMask, currentMaskState);
    }

    @Override
    public String toString() {
        return "UnstackVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",fromIdx=" + from
                        + ",forwardShape=" + forwardShape + ")";
    }
}
