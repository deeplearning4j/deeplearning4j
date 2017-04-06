package org.nd4j.autodiff.graph.vertexfactory;


import org.nd4j.autodiff.graph.api.Vertex;

/**Vertex factory,
 *  used to create nodes from an integer index (0 to nVertices-1 inclusive)
 *  @author Alex Black
 */
public interface VertexFactory<T> {

    /**
     *
     * @param vertexIdx
     * @param args
     * @return
     */
    Vertex<T> create(int vertexIdx, Object[] args);

}
