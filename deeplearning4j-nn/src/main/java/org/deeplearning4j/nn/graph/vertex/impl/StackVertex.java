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
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

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

    private int[][] lastInputShapes;

    public StackVertex(ComputationGraph graph, String name, int vertexIndex, int numInputs) {
        super(graph, name, vertexIndex, numInputs);
    }

    @Override
    public Activations activate(boolean training) {
        // stacking along dimension 0
        // inputs[] is an array of INDArray (e.g.: shape of 3 x [nExamples, nSize])
        // what we want to do is make a stacked output (e.g.: [3 x nExamples, nSize])
        lastInputShapes = null;
        int nStack = input.size();
        int[] inShape = input.get(0).shape();
        int[] outShape = new int[inShape.length];

        // create the new shape
        outShape[0] = nStack * inShape[0];
        for (int i = 1; i < inShape.length; i++) {
            outShape[i] = inShape[i];
        }

        boolean variableLengthTS = false;
        if (inShape.length == 3) {
            //RNN data - check for variable length time series
            int minLength = input.get(0).size(2);
            int maxLength = minLength;
            for (int i = 1; i < input.size(); i++) {
                int thisLength = input.get(i).size(2);
                minLength = Math.min(minLength, thisLength);
                maxLength = Math.max(maxLength, thisLength);
            }
            variableLengthTS = (minLength != maxLength);

            if (!variableLengthTS) {
                return ActivationsFactory.getInstance().create(Nd4j.concat(0, input.getAsArray()));
            }

            outShape[2] = maxLength;
            INDArray out = Nd4j.create(outShape);
            int numExamples = input.get(0).size(0);
            lastInputShapes = new int[input.size()][0];
            for (int i = 0; i < input.size(); i++) {
                out.put(new INDArrayIndex[] {NDArrayIndex.interval(i * numExamples, (i + 1) * numExamples),
                                NDArrayIndex.all(), NDArrayIndex.interval(0, input.get(i).size(2))}, input.get(i));
                lastInputShapes[i] = input.get(i).shape();
            }

            Pair<INDArray, MaskState> p = feedForwardMaskArrays(input.getMaskAsArray(), input.getMaskState(0), getInputMiniBatchSize());
            return ActivationsFactory.getInstance().create(out, p.getFirst(), p.getSecond());
        } else {
            return ActivationsFactory.getInstance().create(Nd4j.concat(0, input.getAsArray()));
        }
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        INDArray epsilon = gradient.get(0);
        // this is basically activate on UnstackVertex
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass");

        if (epsilon == null) {
            //Edge case for stack vertex: stack -> embedding
            //If the null epsilons are a problem in practice, this should be picked up by other layers
            return GradientsFactory.getInstance().create(gradient.getParameterGradients(), new INDArray[input.size()]);
        }

        int nStack = input.size();
        INDArray[] out = new INDArray[nStack];

        int step = epsilon.size(0) / nStack;

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
                                    "Cannot get subset for activations of rank " + input.get(0).rank());
            }
        }

        return GradientsFactory.getInstance().create(null, out);
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
        int size1_ex0 = maskArrays[0].size(1);
        int maxLength = size1_ex0;
        for (int i = 1; i < maskArrays.length; i++) {
            allSameLength &= (size1_ex0 == maskArrays[i].size(1));
            maxLength = Math.max(maxLength, maskArrays[i].size(1));
        }

        if (allSameLength) {
            return new Pair<>(Nd4j.vstack(maskArrays), currentMaskState);
        } else {
            int numExamples = maskArrays[0].size(0);
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
        return "StackVertex(id=" + this.getIndex() + ",name=\"" + this.getName() + ")";
    }
}
