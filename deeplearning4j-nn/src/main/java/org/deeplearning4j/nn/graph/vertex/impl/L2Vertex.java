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
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * L2Vertex calculates the L2 least squares error of two inputs.
 *
 * For example, in Triplet Embedding you can input an anchor and a pos/neg class and use two parallel
 * L2 vertices to calculate two real numbers which can be fed into a LossLayer to calculate TripletLoss.
 *
 * @author Justin Long (crockpotveggies)
 */
public class L2Vertex extends BaseGraphVertex {
    private double eps;

    public L2Vertex(ComputationGraph graph, String name, int vertexIndex, double eps) {
        this(graph, name, vertexIndex, null, null, eps);
    }

    public L2Vertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, double eps) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
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
            throw new IllegalStateException("Cannot do forward pass: input not set");

        INDArray a = inputs[0];
        INDArray b = inputs[1];

        int[] dimensions = new int[a.rank() - 1];
        for (int i = 1; i < a.rank(); i++) {
            dimensions[i - 1] = i;
        }

        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)) {
            return Nd4j.getExecutioner().exec(new EuclideanDistance(a, b), dimensions);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: error not set");

        INDArray a = inputs[0];
        INDArray b = inputs[1];
        INDArray out = doForward(tbptt, workspaceMgr);
        Transforms.max(out, eps, false); // in case of 0

        INDArray dLdlambda = epsilon; //dL/dlambda aka 'epsilon' - from layer above

        INDArray sNegHalf = out.rdiv(1.0); //s^(-1/2) = 1.0 / s^(1/2) = 1.0 / out

        INDArray diff;
        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)){
            diff = a.sub(b);
        }

        INDArray first = dLdlambda.mul(sNegHalf); //Column vector for all cases

        INDArray dLda;
        INDArray dLdb;
        if (a.rank() == 2) {
            //2d case (MLPs etc)
            dLda = diff.muliColumnVector(first);
            try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                dLdb = dLda.neg();
            }
        } else {
            //RNN and CNN case - Broadcast along dimension 0
            dLda = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(diff, first, diff, 0));
            try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATION_GRAD)) {
                dLdb = dLda.neg();
            }
        }

        return new Pair<>(null, new INDArray[] {dLda, dLdb});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString() {
        return "L2Vertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + ")";
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
