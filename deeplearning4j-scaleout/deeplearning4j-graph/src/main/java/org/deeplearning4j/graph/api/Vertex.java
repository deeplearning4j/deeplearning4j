package org.deeplearning4j.graph.api;

import lombok.AllArgsConstructor;

@AllArgsConstructor
public class Vertex<T> {

    private final int idx;
    private final T value;

    public int vertexID() {
        return idx;
    }

    public T getValue() {
        return value;
    }

    @Override
    public String toString() {
        return "vertex(" + idx + "," + (value!=null ? value : "") + ")";
    }
}
