package org.nd4j.autodiff.graph.api;

import lombok.Data;

/** Vertex in a graph
 *
 * @param <T> the opType of the value/object associated with the vertex
 */
@Data
public class Vertex<T> {

    protected  int idx;
    protected int depth;
    protected  T value;

    public Vertex(int idx, int depth, T value) {
        this.idx = idx;
        this.depth = depth;
        if(value == null)
            throw new IllegalArgumentException("No null values allowed");
        this.value = value;
    }

    public void setValue(T value) {
        if(value == null)
            throw new IllegalArgumentException("No null value allowed.");
        this.value = value;
    }

    public int depth() {
        return depth;
    }

    public int vertexID() {
        return idx;
    }

    public T getValue() {
        return value;
    }

}
