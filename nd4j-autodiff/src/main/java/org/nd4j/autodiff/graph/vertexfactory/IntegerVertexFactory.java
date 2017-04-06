package org.nd4j.autodiff.graph.vertexfactory;


import org.nd4j.autodiff.graph.api.Vertex;

public class IntegerVertexFactory implements VertexFactory<Integer> {
    @Override
    public Vertex<Integer> create(int vertexIdx, Object[] args) {
        return new Vertex<>(vertexIdx, vertexIdx);
    }
}
