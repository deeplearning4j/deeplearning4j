package org.deeplearning4j.graph.models;

import org.deeplearning4j.graph.api.Graph;
import org.deeplearning4j.graph.api.Vertex;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;

/**Vectors for nodes in a graph.
 * Provides lookup table and convenience methods for graph vectors
 */
public interface GraphVectors<V,E> extends Serializable {

    public Graph<V,E> getGraph();

    public int numVertices();

    public INDArray getVertexVector(Vertex<V> vertex);

    public INDArray getVertexVector(int vertexIdx);

    public Collection<Vertex<V>> verticesNearest(Vertex<V> vertex, int top);

    double similarity(Vertex<V> vertex1, Vertex<V> vertex2);

    double similarity(int vertexIdx1, int vertexIdx2);

}
