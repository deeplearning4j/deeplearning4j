package org.deeplearning4j.graph.vertexfactory;

import org.deeplearning4j.graph.api.Vertex;

public class IntegerVertexFactory implements VertexFactory<Integer> {
    @Override
    public Vertex<Integer> create(int vertexIdx) {
        return new Vertex<>(vertexIdx, vertexIdx);
    }
}
