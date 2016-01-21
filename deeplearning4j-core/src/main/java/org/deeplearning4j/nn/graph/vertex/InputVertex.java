package org.deeplearning4j.nn.graph.vertex;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public class InputVertex extends BaseGraphVertex {


    public InputVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, null, outputVertices);
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
    public boolean isInputVertex(){
        return true;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {
        throw new UnsupportedOperationException("Cannot do forward pass for InputVertex");
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, int tbpttBackwardLength) {
        throw new UnsupportedOperationException("Cannot do backward pass for InputVertex");
    }
}
