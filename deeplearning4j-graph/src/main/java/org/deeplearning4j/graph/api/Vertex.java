package org.deeplearning4j.graph.api;

import lombok.AllArgsConstructor;

/** Vertex in a graph
 *
 * @param <T> the type of the value/object associated with the vertex
 */
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
        return "vertex(" + idx + "," + (value != null ? value : "") + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Vertex))
            return false;
        Vertex<?> v = (Vertex<?>) o;
        if (idx != v.idx)
            return false;
        if ((value == null && v.value != null) || (value != null && v.value == null))
            return false;
        return value == null || value.equals(v.value);
    }

    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + idx;
        result = 31 * result + (value == null ? 0 : value.hashCode());
        return result;
    }
}
