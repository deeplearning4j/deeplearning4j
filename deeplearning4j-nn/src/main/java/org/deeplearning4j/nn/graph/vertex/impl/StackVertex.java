/*
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

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

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

    public StackVertex(ComputationGraph graph, String name, int vertexIndex){
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
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {
        // stacking along dimension 0
        // inputs[] is an array of INDArray (e.g.: shape of 3 x [nExamples, nSize])
        // what we want to do is make a stacked output (e.g.: [3 x nExamples, nSize])
        int nStack = inputs.length;
        int[] inShape = inputs[0].shape();
        int[] outShape = new int[inShape.length];

        // create the new shape
        for ( int i=0; i<inShape.length; i++ ) {
            if(i==0) outShape[0] = nStack * inShape[0];
            else outShape[i] = inShape[i];
        }

        INDArray out = Nd4j.create(outShape);

        //Simplest case: no masking arrays, all same length
        // loop through indexes for 2D, 3D, 4D...
        INDArrayIndex[] indexes = new INDArrayIndex[inShape.length];
        for (int i=0; i<inShape.length; i++) {
            indexes[i] = NDArrayIndex.all();
        }

        int rowCount = 0;
        for (INDArray input : inputs) {
            int nEx = input.size(0);
            indexes[0] = NDArrayIndex.interval(rowCount, rowCount + nEx);
            out.put(indexes, input);
            rowCount += nEx;
        }

        return out;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        // this is basically doForward on UnstackVertex
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: input not set");

        int nStack = inputs.length;
        INDArray[] out = new INDArray[nStack];

        int step = epsilon.size(0)/nStack;

        for(int i=0; i<nStack; i++ ) {
            switch (epsilon.rank()) {
                case 2:
                    out[i] = epsilon.get(NDArrayIndex.interval(i*step, (i+1)*step), NDArrayIndex.all());
                    break;
                case 3:
                    out[i] = epsilon.get(NDArrayIndex.interval(i*step, (i+1)*step), NDArrayIndex.all(), NDArrayIndex.all());
                    break;
                case 4:
                    out[i] = epsilon.get(NDArrayIndex.interval(i*step, (i+1)*step), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                    break;
                default:
                    throw new UnsupportedOperationException("Cannot get subset for activations of rank " + inputs[0].rank());
            }
        }

        return new Pair<>(null,out);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        //Cases here: no mask arrays, or all mask arrays - all of the same size
        if(maskArrays == null ){
            return new Pair<>(null, currentMaskState);
        }

        // stacking along dimension 0
        //Given masks are all either 1d (column vector) or 2d (examples, timeSeriesLength) we can just vStack the masks

        return new Pair<>(Nd4j.vstack(maskArrays), currentMaskState);
    }

    @Override
    public String toString(){
        return "StackVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + ")";
    }
}
