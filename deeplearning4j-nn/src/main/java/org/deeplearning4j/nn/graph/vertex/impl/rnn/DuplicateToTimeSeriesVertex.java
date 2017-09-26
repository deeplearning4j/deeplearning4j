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

package org.deeplearning4j.nn.graph.vertex.impl.rnn;

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
 * DuplicateToTimeSeriesVertex is a vertex that goes from 2d activations to a 3d time series activations, by means of
 * duplication. That is, given a 2d input with shape [numExamples,nIn] duplicate each row to give output of
 * [numExamples,nIn,timeSeriesLength], where the activations are the same for all time steps.<br>
 * <br>
 * To use DuplicateToTimeSeriesVertex, a user must specify 2 inputs (any order is OK)
 * (a) A 2d input, to be duplicated, and<br>
 * (b) A 3d (time series) input, that provides the shape/length, and a mask array (if present) for handling variable-
 *     length time series.<br>
 * This method is used for example in some sequence to sequence models.<br>
 * @author Alex Black
 */
public class DuplicateToTimeSeriesVertex extends BaseGraphVertex {

    private int tsInputIdx = -1;

    public DuplicateToTimeSeriesVertex(String name, int vertexIndex, int numInputs) {
        super(name, vertexIndex, numInputs);

    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Activations activate(boolean training) {
        if(input == null || input.anyActivationsNull()){
            throw new IllegalStateException("Cannot perform forward pass: input is not set");
        }

        INDArray act2d = null;
        INDArray act3d = null;
        INDArray mask = null;
        MaskState maskState = null;
        if(input.get(0).rank() == 2){
            act2d = input.get(0);
        } else if(input.get(0).rank() == 3){
            act3d = input.get(0);
            mask = input.getMask(0);
            maskState = input.getMaskState(0);
            tsInputIdx = 0;
        }
        if(input.get(1).rank() == 2){
            act2d = input.get(1);
        } else if(input.get(1).rank() == 3){
            act3d = input.get(1);
            mask = input.getMask(1);
            maskState = input.getMaskState(1);
            tsInputIdx = 1;
        }

        if(act2d == null || act3d == null){
            throw new IllegalStateException("Expected to get 1x rank 2 array (to duplicate) and 1x rank 3 array" +
                    " (to determine output size). Got arrays of ranks " + input.get(0).rank() + " and "
                    + input.get(1).rank() + ", with shapes " + Arrays.toString(input.get(0).shape()) + " and "
                    + Arrays.toString(input.get(1).shape()));
        }

        int tsLength = act3d.size(2);
        int[] outShape = new int[] {act2d.size(0), act2d.size(1), tsLength};
        INDArray out = Nd4j.create(outShape);
        //TODO: replace with broadcast once ND4J #2066 is closed
        for (int i = 0; i < tsLength; i++) {
            out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i)}, act2d);
        }

        return ActivationsFactory.getInstance().create(out, mask, maskState);
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        //Because we duplicated for each time step: simply need to sum along time for errors/epsilons
        //Note that we don't need to worry about masks here: if masks were relevant, they should be applied already
        // in the activations gradients that are coming as input from the layer/vertex above

        INDArray outEps3d = Nd4j.create(input.get(tsInputIdx).shape());
        if(tsInputIdx == 0){
            return GradientsFactory.getInstance().createPair(outEps3d, gradient.get(0).sum(2), null);
        } else {
            return GradientsFactory.getInstance().createPair(gradient.get(0).sum(2), outEps3d, null);
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return "DuplicateToTimeSeriesVertex()";
    }
}
