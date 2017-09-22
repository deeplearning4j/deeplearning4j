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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

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
    private int forwardShape[];
    private int step;

    public UnstackVertex(String name, int vertexIndex, int numInputs, int from, int stackSize) {
        super(name, vertexIndex, numInputs);
        this.from = from;
        this.stackSize = stackSize;
    }

    @Override
    public Activations activate(boolean training) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: input not set");

        // once we know the inputs, save the shape and interval size for doBackward
        this.forwardShape = Arrays.copyOf(input.get(0).shape(), input.get(0).rank());
        this.step = input.get(0).size(0) / stackSize;
        int start = from * step;
        int end = (from + 1) * step;

        INDArray ret;
        switch (input.get(0).rank()) { //TODO remove the dups here if/when possible (gradient checks must pass)
            case 2:
                ret = input.get(0).get(NDArrayIndex.interval(start, end), NDArrayIndex.all()).dup();
                break;
            case 3:
                ret = input.get(0).get(NDArrayIndex.interval(start, end), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                break;
            case 4:
                ret = input.get(0).get(NDArrayIndex.interval(start, end), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all()).dup();
                break;
            default:
                throw new UnsupportedOperationException(
                                "Cannot get subset for activations of rank " + input.get(0).rank());
        }

        Pair<INDArray,MaskState> p = feedForwardMaskArrays(input.getMaskAsArray(), input.getMaskState(0), getInputMiniBatchSize());

        return ActivationsFactory.getInstance().create(ret, p.getFirst(), p.getSecond());
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: error not set");
        INDArray epsilon = gradient.get(0);

        INDArray out = Nd4j.zeros(forwardShape);
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
        return GradientsFactory.getInstance().create(out, null);
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
        int mb = maskArrays[0].size(0) / stackSize;
        int start = from * mb;
        int end = (from + 1) * mb;
        INDArray outMask = maskArrays[0].get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
        return new Pair<>(outMask, currentMaskState);
    }

    @Override
    public String toString() {
        return "UnstackVertex(id=" + this.getIndex() + ",name=\"" + this.getName() + "\",fromIdx=" + from
                        + ",forwardShape=" + forwardShape + ")";
    }
}
