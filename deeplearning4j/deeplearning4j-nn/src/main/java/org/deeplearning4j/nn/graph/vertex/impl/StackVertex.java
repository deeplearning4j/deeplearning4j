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

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * StackVertex allows for stacking of inputs so that they may be forwarded through
 * a network. This is useful for cases such as Triplet Embedding, where shared parameters
 * are not supported by the network.
 *
 * This vertex will automatically stack all available inputs.
 *
 * @author Justin Long (crockpotveggies)
 */
public class StackVertex extends BaseGraphVertex {

    private long[][] lastInputShapes;

    public StackVertex(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, null, null);
    }

    public StackVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
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
        // stacking along dimension 0
        // inputs[] is an array of INDArray (e.g.: shape of 3 x [nExamples, nSize])
        // what we want to do is make a stacked output (e.g.: [3 x nExamples, nSize])
        lastInputShapes = null;
        int nStack = inputs.length;
        val inShape = inputs[0].shape();
        val outShape = new long[inShape.length];

        // create the new shape
        outShape[0] = nStack * inShape[0];
        for (int i = 1; i < inShape.length; i++) {
            outShape[i] = inShape[i];
        }

        boolean variableLengthTS = false;
        if (inShape.length == 3) {
            //RNN data - check for variable length time series
            long minLength = inputs[0].size(2);
            long maxLength = minLength;
            for (int i = 1; i < inputs.length; i++) {
                long thisLength = inputs[i].size(2);
                minLength = Math.min(minLength, thisLength);
                maxLength = Math.max(maxLength, thisLength);
            }
            variableLengthTS = (minLength != maxLength);

            if (!variableLengthTS) {
                try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)) {
                    return Nd4j.concat(0, inputs);
                }
            }

            outShape[2] = maxLength;
            INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, outShape);
            long numExamples = inputs[0].size(0);
            lastInputShapes = new long[inputs.length][0];
            for (int i = 0; i < inputs.length; i++) {
                out.put(new INDArrayIndex[] {NDArrayIndex.interval(i * numExamples, (i + 1) * numExamples),
                                NDArrayIndex.all(), NDArrayIndex.interval(0, inputs[i].size(2))}, inputs[i]);
                lastInputShapes[i] = inputs[i].shape();
            }

            return out;
        } else {
            try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)) {
                return Nd4j.concat(0, inputs);
            }
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        // this is basically doForward on UnstackVertex
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: input not set");

        if (epsilon == null) {
            //Edge case for stack vertex: stack -> embedding
            //If the null epsilons are a problem in practice, this should be picked up by other layers
            return new Pair<>(null, new INDArray[inputs.length]);
        }

        int nStack = inputs.length;
        INDArray[] out = new INDArray[nStack];

        long step = epsilon.size(0) / nStack;

        for (int i = 0; i < nStack; i++) {
            switch (epsilon.rank()) {
                case 2:
                    out[i] = epsilon.get(NDArrayIndex.interval(i * step, (i + 1) * step), NDArrayIndex.all());
                    break;
                case 3:
                    if (lastInputShapes != null) {
                        //Variable length time series case
                        out[i] = epsilon.get(NDArrayIndex.interval(i * step, (i + 1) * step), NDArrayIndex.all(),
                                        NDArrayIndex.interval(0, lastInputShapes[i][2]));
                    } else {
                        out[i] = epsilon.get(NDArrayIndex.interval(i * step, (i + 1) * step), NDArrayIndex.all(),
                                        NDArrayIndex.all());
                    }
                    break;
                case 4:
                    out[i] = epsilon.get(NDArrayIndex.interval(i * step, (i + 1) * step), NDArrayIndex.all(),
                                    NDArrayIndex.all(), NDArrayIndex.all());
                    break;
                default:
                    throw new UnsupportedOperationException(
                                    "Cannot get subset for activations of rank " + inputs[0].rank());
            }
        }

        for( int i=0; i<nStack; i++ ){
            out[i] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, out[i]);
        }

        return new Pair<>(null, out);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                    int minibatchSize) {
        //Cases here: no mask arrays, or all mask arrays - all of the same size
        if (maskArrays == null) {
            return new Pair<>(null, currentMaskState);
        }

        // stacking along dimension 0
        //Given masks are all either 1d (column vector) or 2d (examples, timeSeriesLength) we can just vStack the masks
        //However: variable length TS might have different length masks...
        boolean allSameLength = true;
        long size1_ex0 = maskArrays[0].size(1);
        long maxLength = size1_ex0;
        for (int i = 1; i < maskArrays.length; i++) {
            allSameLength &= (size1_ex0 == maskArrays[i].size(1));
            maxLength = Math.max(maxLength, maskArrays[i].size(1));
        }

        if (allSameLength) {
            return new Pair<>(Nd4j.vstack(maskArrays), currentMaskState);
        } else {
            long numExamples = maskArrays[0].size(0);
            INDArray outMask = Nd4j.create(maskArrays.length * numExamples, maxLength);
            for (int i = 0; i < maskArrays.length; i++) {
                outMask.put(new INDArrayIndex[] {NDArrayIndex.interval(i * numExamples, (i + 1) * numExamples),
                                NDArrayIndex.interval(0, maskArrays[i].size(1))}, maskArrays[i]);
            }

            return new Pair<>(outMask, currentMaskState);
        }
    }

    @Override
    public String toString() {
        return "StackVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + ")";
    }
}
