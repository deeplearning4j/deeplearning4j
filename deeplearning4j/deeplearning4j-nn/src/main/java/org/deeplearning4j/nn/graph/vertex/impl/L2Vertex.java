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
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

public class L2Vertex extends BaseGraphVertex {
    private double eps;

    public L2Vertex(ComputationGraph graph, String name, int vertexIndex, double eps, DataType dataType) {
        this(graph, name, vertexIndex, null, null, eps, dataType);
    }

    public L2Vertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, double eps, DataType dataType) {
        super(graph, name, vertexIndex, inputVertices, outputVertices, dataType);
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

        long[] dimensions = new long[a.rank() - 1];
        for (int i = 1; i < a.rank(); i++) {
            dimensions[i - 1] = i;
        }


        INDArray arr = Nd4j.getExecutioner().exec(new EuclideanDistance(a, b, dimensions));
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS,arr.reshape(arr.size(0), 1));
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
        diff = a.sub(b);


        INDArray first = dLdlambda.mul(sNegHalf); //Column vector for all cases

        INDArray dLda;
        INDArray dLdb;
        if (a.rank() == 2) {
            //2d case (MLPs etc)
            dLda = diff.muliColumnVector(first);
            dLdb = dLda.neg();

        } else {
            //RNN and CNN case - Broadcast along dimension 0
            dLda = Nd4j.getExecutioner().exec(new BroadcastMulOp(diff, first, diff, 0));
            dLdb = dLda.neg();

        }

        return new Pair<>(null, new INDArray[] {workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD,dLda), workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD,dLdb)});
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
