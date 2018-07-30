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
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * L2NormalizeVertex performs L2 normalization on a single input.
 *
 * @author Justin Long (crockpotveggies)
 * @author Alex Black (AlexDBlack)
 */
public class L2NormalizeVertex extends BaseGraphVertex {

    private static final int[] DEFAULT_RANK2_DIMS = new int[] {1};
    private static final int[] DEFAULT_RANK3_DIMS = new int[] {1, 2};
    private static final int[] DEFAULT_RANK4_DIMS = new int[] {1, 2, 3};

    private int[] dimension;
    private double eps;

    public L2NormalizeVertex(ComputationGraph graph, String name, int vertexIndex, int[] dimension, double eps) {
        this(graph, name, vertexIndex, null, null, dimension, eps);
    }

    public L2NormalizeVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, int[] dimension, double eps) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.dimension = dimension;
        this.eps = eps;
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
            throw new IllegalStateException("Cannot do forward pass: inputs not set (L2NormalizeVertex " + vertexName
                            + " idx " + vertexIndex + ")");

        // L2 norm along all dimensions except 0, unless user-specified
        // x / |x|2
        INDArray x = inputs[0];
        int[] dimensions = getDimensions(x);

        INDArray xNorm2 = x.norm2(dimensions);
        Transforms.max(xNorm2, eps, false);
        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)){
            if (x.rank() == 2) {
                return x.divColumnVector(xNorm2);
            } else {
                INDArray out = Nd4j.createUninitialized(x.shape(), x.ordering());
                return Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(x, xNorm2, out, 0));
            }
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set (L2NormalizeVertex " + vertexName
                            + " idx " + vertexIndex + ")");

        INDArray x = inputs[0];
        int[] dimensions = getDimensions(x);

        INDArray norm = x.norm2(dimensions);
        INDArray norm3 = Transforms.pow(norm, 3.0, true);
        Transforms.max(norm, eps, false); // in case of div/0
        Transforms.max(norm3, eps, false);

        INDArray dLdx;
        if (x.rank() == 2) {
            // 2D case
            try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                dLdx = epsilon.divColumnVector(norm);
            }
            INDArray xDivNorm3 = x.divColumnVector(norm3);
            dLdx.subi(xDivNorm3.muliColumnVector(epsilon.mul(x).sum(1)));
        } else {
            //RNN and CNN case - Broadcast along dimension 0
            INDArray dx = epsilon.mul(x).sum(dimensions);

            //x / |x|_2^3 * sum_k (dLda*x)
            INDArray xDivNorm3 = Nd4j.createUninitialized(x.shape(), x.ordering());
            Nd4j.getExecutioner().exec(new BroadcastDivOp(x, norm3, xDivNorm3, 0));
            Nd4j.getExecutioner().exec(new BroadcastMulOp(xDivNorm3, dx, xDivNorm3, 0));

            //1/|x|_2 * dLda - above
            dLdx = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, epsilon.shape(), epsilon.ordering());
            Nd4j.getExecutioner().exec(new BroadcastDivOp(epsilon, norm, dLdx, 0));
            dLdx.subi(xDivNorm3);
        }

        return new Pair<>(null, new INDArray[] {dLdx});
    }

    private int[] getDimensions(INDArray x) {
        if (dimension == null || dimension.length < 1) {
            switch (x.rank()) {
                case 2:
                    return DEFAULT_RANK2_DIMS;
                case 3:
                    return DEFAULT_RANK3_DIMS;
                case 4:
                    return DEFAULT_RANK4_DIMS;
                default:
                    throw new RuntimeException();
            }
        }
        return dimension;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
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

    @Override
    public String toString() {
        return "L2NormalizeVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + ",dim=\""
                        + dimension + "\")";
    }
}
