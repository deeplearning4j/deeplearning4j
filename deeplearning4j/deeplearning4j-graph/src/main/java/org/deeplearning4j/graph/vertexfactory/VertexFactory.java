package org.deeplearning4j.graph.vertexfactory;

import org.deeplearning4j.graph.api.Vertex;

/**Vertex factory, used to create nodes from an integer index (0 to nVertices-1 inclusive)
 */
public interface VertexFactory<T> {

    Vertex<T> create(int vertexIdx);

}
