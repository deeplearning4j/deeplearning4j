package org.deeplearning4j.nn.graph.vertex;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class SubsetVertex extends BaseGraphVertex {

    private int from;
    private int to;
    private int[] forwardShape;

    public SubsetVertex(ComputationGraph graph, String name, int vertexIndex, int from, int to){
        this(graph,name,vertexIndex,null,null,from,to);
    }

    public SubsetVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices,
                        int from, int to) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.from = from;
        this.to = to;
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

        forwardShape = Arrays.copyOf(inputs[0].shape(), inputs[0].rank());

        switch (inputs[0].rank()) {
            case 2:
                return inputs[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true));
            case 3:
                return inputs[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all());
            case 4:
                return inputs[0].get(NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all(), NDArrayIndex.all());
            default:
                throw new UnsupportedOperationException("Cannot get subset for activations of rank " + inputs[0].rank());
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, int tbpttBackwardLength) {
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: error not set");

        INDArray out = Nd4j.zeros(forwardShape);
        switch (forwardShape.length) {
            case 2:
                out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(from, to, true)}, epsilons[0]);
                break;
            case 3:
                out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all()}, epsilons[0]);
                break;
            case 4:
                out.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(from, to, true), NDArrayIndex.all(), NDArrayIndex.all()}, epsilons[0]);
                break;
            default:
                throw new RuntimeException("Invalid activation rank");  //Should never happen
        }
        return new Pair<>(null,new INDArray[]{out});

    }
}
