package org.deeplearning4j.graph.api;

import org.deeplearning4j.graph.exception.NoEdgesException;

import java.util.List;
import java.util.Random;

public interface Graph<V,E> {

    public int numVertices();

    public Vertex<V> getVertex(int idx);

    public List<Vertex<V>> getVertices(int[] indexes);

    public List<Vertex<V>> getVertices(int from, int to);

    public void addEdge(Edge<E> edge);

    public void addEdge(int from, int to, E value, boolean directed);

    public List<Edge<E>> getEdgesOut(int vertex);

    public int getNumEdgesOut(int vertex);

    public Vertex<V> getRandomConnectedVertex(int vertex, Random rng) throws NoEdgesException;

    public List<Vertex<V>> getConnectedVertices(int vertex);

    public int[] getConnectedVertexIndices(int vertex);


}
