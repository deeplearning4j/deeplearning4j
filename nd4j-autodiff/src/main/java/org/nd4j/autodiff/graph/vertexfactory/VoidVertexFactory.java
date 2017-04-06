package org.nd4j.autodiff.graph.vertexfactory;


import org.nd4j.autodiff.graph.api.Vertex;

public class VoidVertexFactory implements VertexFactory<Void> {
    @Override
    public Vertex<Void> create(int vertexIdx, Object[] args) {
        return new Vertex<>(vertexIdx, null);
    }
}
