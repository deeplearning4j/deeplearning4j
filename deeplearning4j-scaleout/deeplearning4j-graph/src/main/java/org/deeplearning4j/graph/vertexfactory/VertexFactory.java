package org.deeplearning4j.graph.vertexfactory;

import org.deeplearning4j.graph.api.Vertex;

/**
 * Created by Alex on 9/11/2015.
 */
public interface VertexFactory<T> {

    public Vertex<T> create(int vertexIdx);

}
