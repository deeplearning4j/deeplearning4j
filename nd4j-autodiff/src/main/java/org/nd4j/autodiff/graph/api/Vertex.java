package org.nd4j.autodiff.graph.api;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

/** Vertex in a graph
 *
 * @param <T> the type of the value/object associated with the vertex
 */
@AllArgsConstructor
@Data
@Builder
public class Vertex<T> {

    protected final int idx;
    protected final T value;

    public int vertexID() {
        return idx;
    }

    public T getValue() {
        return value;
    }

}
