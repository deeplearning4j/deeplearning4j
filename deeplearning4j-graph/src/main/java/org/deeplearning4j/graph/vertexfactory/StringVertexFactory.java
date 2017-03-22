package org.deeplearning4j.graph.vertexfactory;

import org.deeplearning4j.graph.api.Vertex;

public class StringVertexFactory implements VertexFactory<String> {

    private final String format;

    public StringVertexFactory() {
        this(null);
    }

    public StringVertexFactory(String format) {
        this.format = format;
    }

    @Override
    public Vertex<String> create(int vertexIdx) {
        if (format != null)
            return new Vertex<>(vertexIdx, String.format(format, vertexIdx));
        else
            return new Vertex<>(vertexIdx, String.valueOf(vertexIdx));
    }
}
