package org.deeplearning4j.graph.models;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.Vertex;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**Vectors for nodes in a graph.
 * Provides lookup table and convenience methods for graph vectors
 */
public interface GraphVectors<V, E> extends Serializable {

    public IGraph<V, E> getGraph();

    public int numVertices();

    public int getVectorSize();

    public INDArray getVertexVector(Vertex<V> vertex);

    public INDArray getVertexVector(int vertexIdx);

    public int[] verticesNearest(int vertexIdx, int top);

    double similarity(Vertex<V> vertex1, Vertex<V> vertex2);

    double similarity(int vertexIdx1, int vertexIdx2);

}
