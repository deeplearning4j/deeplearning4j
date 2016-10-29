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
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * L2Vertex calculates the L2
 *
 * @author Justin Long (crockpotveggies)
 */
public class L2Vertex extends BaseGraphVertex {

    private String inputName;
    private int inputVertexIndex;

    public L2Vertex(ComputationGraph graph, String name, int vertexIndex, String inputVertexName){
        this(graph,name,vertexIndex,null,null,inputVertexName);
    }

    public L2Vertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                    VertexIndices[] outputVertices, String inputName) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.inputName = inputName;
        this.inputVertexIndex = graph.getConfiguration().getNetworkInputs().indexOf(inputName);
        if(inputVertexIndex == -1)  throw new IllegalArgumentException("Invalid input name: \"" + inputName + "\" not found in list "
                + "of network inputs (" + graph.getConfiguration().getNetworkInputs() + ")");
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
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: input not set");

        INDArray out = Transforms.sqrt(Transforms.pow(inputs[0].sub(inputs[1]), 2));

        return out;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: error not set");
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public String toString(){
        return "L2Vertex(inputName=" + inputName + ")";
    }
}
