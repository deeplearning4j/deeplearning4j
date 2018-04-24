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
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * A ScaleVertex is used to scale the size of activations of a single layer<br>
 * For example, ResNet activations can be scaled in repeating blocks to keep variance
 * under control.
 *
 * @author Justin Long (@crockpotveggies)
 */
public class ScaleVertex extends BaseGraphVertex {

    private double scaleFactor;

    public ScaleVertex(ComputationGraph graph, String name, int vertexIndex, double scaleFactor) {
        this(graph, name, vertexIndex, null, null, scaleFactor);
    }

    public ScaleVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, double scaleFactor) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.scaleFactor = scaleFactor;
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
            throw new IllegalStateException("Cannot do forward pass: inputs not set (ScaleVertex " + vertexName
                            + " idx " + vertexIndex + ")");

        if (inputs.length > 1)
            throw new IllegalArgumentException(
                            "ScaleVertex (name " + vertexName + " idx " + vertexIndex + ") only supports 1 input.");

        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)){
            return inputs[0].mul(scaleFactor);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set (ScaleVertex " + vertexName
                            + " idx " + vertexIndex + ")");

        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)){
            return new Pair<>(null, new INDArray[] {epsilon.mul(scaleFactor)});
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException(
                            "Vertex does not have gradients; gradients view array cannot be set here (ScaleVertex "
                                            + vertexName + " idx " + vertexIndex + ")");
    }

    @Override
    public String toString() {
        return "ScaleVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",scaleFactor="
                        + scaleFactor + ")";
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        //No op
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return new Pair<>(maskArrays[0], currentMaskState);
    }
}
