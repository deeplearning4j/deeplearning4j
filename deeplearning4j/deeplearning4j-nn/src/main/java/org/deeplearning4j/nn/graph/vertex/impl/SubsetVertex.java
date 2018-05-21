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

/** SubsetVertex is used to select a subset of the activations out of another GraphVertex.<br>
 * For example, a subset of the activations out of a layer.<br>
 * Note that this subset is specifying by means of an interval of the original activations.
 * For example, to get the first 10 activations of a layer (or, first 10 features out of a CNN layer) use
 * new SubsetVertex(0,9).<br>
 * In the case of convolutional (4d) activations, this is done along channels.
 * @author Alex Black
 */
public class SubsetVertex extends BaseGraphVertex {
    private int from;
    private int to; //inclusive
    private long[] forwardShape;

    public SubsetVertex(ComputationGraph graph, String name, int vertexIndex, int from, int to) {
        this(graph, name, vertexIndex, null, null, from, to);
    }

    public SubsetVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, int from, int to) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.from = from;
        this.to = to;
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

        forwardShape = Arrays.copyOf(inputs[0].shape(), inputs[0].rank());

        INDArray out;
        switch (inputs[0].rank()) {
            case 2:
                out = inputs[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true));
                break;
            case 3:
                out = inputs[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all());
                break;
            case 4:
                out = inputs[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all(),
                                NDArrayIndex.all());
                break;
            default:
                throw new UnsupportedOperationException(
                                "Cannot get subset for activations of rank " + inputs[0].rank());
        }
        return workspaceMgr.dup(ArrayType.ACTIVATIONS, out);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: error not set");

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, forwardShape);
        switch (forwardShape.length) {
            case 2:
                out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(from, to, true)}, epsilon);
                break;
            case 3:
                out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(from, to, true),
                                NDArrayIndex.all()}, epsilon);
                break;
            case 4:
                out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.interval(from, to, true),
                                NDArrayIndex.all(), NDArrayIndex.all()}, epsilon);
                break;
            default:
                throw new RuntimeException("Invalid activation rank"); //Should never happen
        }
        return new Pair<>(null, new INDArray[] {out});
    }

    @Override
    public String toString() {
        return "SubsetVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",fromIdx=" + from
                        + ",toIdx=" + to + ")";
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        //No op: subset just provides part of the activations for each example (or time step)
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return new Pair<>(maskArrays[0], currentMaskState);
    }
}
