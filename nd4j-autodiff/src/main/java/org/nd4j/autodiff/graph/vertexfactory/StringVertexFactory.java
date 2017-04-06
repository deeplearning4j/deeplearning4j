package org.nd4j.autodiff.graph.vertexfactory;


import org.nd4j.autodiff.graph.api.Vertex;

public class StringVertexFactory implements VertexFactory<String> {

    private final String format;

    public StringVertexFactory() {
        this(null);
    }

    public StringVertexFactory(String format) {
        this.format = format;
    }

    @Override
    public Vertex<String> create(int vertexIdx, Object[] args) {
        if (format != null)
            return new Vertex<>(vertexIdx, String.format(format, vertexIdx));
        else
            return new Vertex<>(vertexIdx, String.valueOf(vertexIdx));
    }
}
