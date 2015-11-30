package org.deeplearning4j.graph.api;

import java.util.Iterator;

/**Represents a sequence of vertices in a graph.<br>
 * General-purpose, but can be used to represent a walk on a graph, for example.
 */
public interface IVertexSequence<T> extends Iterator<Vertex<T>> {

    /** Length of the vertex sequence */
    int sequenceLength();

}
