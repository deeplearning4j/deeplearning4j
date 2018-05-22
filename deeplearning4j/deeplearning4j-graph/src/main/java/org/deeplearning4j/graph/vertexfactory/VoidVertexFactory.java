package org.deeplearning4j.graph.vertexfactory;

import org.deeplearning4j.graph.api.Vertex;

public class VoidVertexFactory implements VertexFactory<Void> {
    @Override
    public Vertex<Void> create(int vertexIdx) {
        return new Vertex<>(vertexIdx, null);
    }
}
