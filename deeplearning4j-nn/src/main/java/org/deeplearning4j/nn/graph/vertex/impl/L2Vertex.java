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

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

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

    public L2Vertex(String name, int vertexIndex, int numInputs, double eps) {
        super(name, vertexIndex, numInputs);
        this.eps = eps;
    }

    @Override
    public Activations activate(boolean training) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: input not set");

        INDArray a = input.get(0);
        INDArray b = input.get(1);

        int[] dimensions = new int[a.rank() - 1];
        for (int i = 1; i < a.rank(); i++) {
            dimensions[i - 1] = i;
        }

        return ActivationsFactory.getInstance().create(Nd4j.getExecutioner().exec(new EuclideanDistance(a, b), dimensions));
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: error not set");
        INDArray epsilon = gradient.get(0);

        INDArray a = input.get(0);
        INDArray b = input.get(1);
        INDArray out = activate(true).get(0);
        Transforms.max(out, eps, false); // in case of 0

        INDArray dLdlambda = epsilon; //dL/dlambda aka 'epsilon' - from layer above

        INDArray sNegHalf = out.rdiv(1.0); //s^(-1/2) = 1.0 / s^(1/2) = 1.0 / out

        INDArray diff = a.sub(b);

        INDArray first = dLdlambda.mul(sNegHalf); //Column vector for all cases

        INDArray dLda;
        INDArray dLdb;
        if (a.rank() == 2) {
            //2d case (MLPs etc)
            dLda = diff.muliColumnVector(first);
            dLdb = dLda.neg();
        } else {
            //RNN and CNN case - Broadcast along dimension 0
            dLda = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(diff, first, diff, 0));
            dLdb = dLda.neg();
        }

        return GradientsFactory.getInstance().createPair(dLda, dLdb, null);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString() {
        return "L2Vertex(id=" + this.getIndex() + ",name=\"" + this.getName() + ")";
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
