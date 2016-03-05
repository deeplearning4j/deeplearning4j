package org.deeplearning4j.models.sequencevectors.graph.vertex;


import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**Vertex factory, used to create nodes from an integer index (0 to nVertices-1 inclusive)
 */
public interface VertexFactory<T extends SequenceElement> {

    Vertex<T> create(int vertexIdx);

    Vertex<T> create(T element);

    Vertex<T> create(int vertexIdx, T element);
}
