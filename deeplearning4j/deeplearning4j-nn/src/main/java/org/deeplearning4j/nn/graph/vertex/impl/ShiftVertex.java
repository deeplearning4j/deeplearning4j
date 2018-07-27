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
 * A ShiftVertex is used to shift the activations of a single layer<br>
 * One could use it to add a bias or as part of some other calculation.
 * For example, Highway Layers need them in two places. One, it's often
 * useful to have the gate weights have a large negative bias. (Of course
 * for this, we could just initialize the biases that way.)
 * But, _also_ it needs to do this:
 * (1-sigmoid(weight * input + bias)) (*) input + sigmoid(weight * input + bias) (*) activation(w2 * input + bias) ((*) is hadamard product)
 * So, here, we could have
 * 1. a DenseLayer that does the sigmoid
 * 2. a ScaleVertex(-1) and
 * 3. a ShiftVertex(1)
 * to accomplish that.
 *
 * @author Binesh Bannerjee (binesh_binesh@hotmail.com, @bnsh on gitter)
 */
public class ShiftVertex extends BaseGraphVertex {

    private double shiftFactor;

    public ShiftVertex(ComputationGraph graph, String name, int vertexIndex, double shiftFactor) {
        this(graph, name, vertexIndex, null, null, shiftFactor);
    }

    public ShiftVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, double shiftFactor) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.shiftFactor = shiftFactor;
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
            throw new IllegalStateException("Cannot do forward pass: inputs not set (ShiftVertex " + vertexName
                            + " idx " + vertexIndex + ")");

        if (inputs.length > 1)
            throw new IllegalArgumentException(
                            "ShiftVertex (name " + vertexName + " idx " + vertexIndex + ") only supports 1 input.");

        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)){
            return inputs[0].add(shiftFactor);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set (ShiftVertex " + vertexName
                            + " idx " + vertexIndex + ")");

        return new Pair<>(null, new INDArray[] {workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon)});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException(
                            "Vertex does not have gradients; gradients view array cannot be set here (ShiftVertex "
                                            + vertexName + " idx " + vertexIndex + ")");
    }

    @Override
    public String toString() {
        return "ShiftVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",shiftFactor="
                        + shiftFactor + ")";
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
