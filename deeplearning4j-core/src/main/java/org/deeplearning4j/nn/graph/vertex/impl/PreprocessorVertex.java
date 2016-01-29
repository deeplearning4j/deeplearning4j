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
}
