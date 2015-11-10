package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Vertex;

import java.io.IOException;
import java.util.List;

public interface VertexLoader<V> {

    List<Vertex<V>> loadVertices(String path) throws IOException;

}
