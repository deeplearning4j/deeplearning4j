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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;

/** PreprocessorVertex is a simple adaptor class that allows a {@link InputPreProcessor} to be used in a ComputationGraph
 * GraphVertex, without it being associated with a layer.
 * @author Alex Black
 */
public class PreprocessorVertex extends BaseGraphVertex {

    private InputPreProcessor preProcessor;

    public PreprocessorVertex(ComputationGraph graph, String name, int vertexIndex, InputPreProcessor preProcessor) {
        this(graph, name, vertexIndex, null, null, preProcessor);
    }

    public PreprocessorVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                                 VertexIndices[] outputVertices, InputPreProcessor preProcessor) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.preProcessor = preProcessor;
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
        return preProcessor.preProcess(inputs[0],graph.batchSize());
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt) {
        return new Pair<>(null,new INDArray[]{preProcessor.backprop(epsilons[0],graph.batchSize())});
    }

    @Override
    public String toString() {
        return "PreprocessorVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\",preProcessor=" + preProcessor.toString() + ")";
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }
}
