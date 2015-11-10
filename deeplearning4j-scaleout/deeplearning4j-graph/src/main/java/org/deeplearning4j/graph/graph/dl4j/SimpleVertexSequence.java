package org.deeplearning4j.graph.graph.dl4j;

import org.deeplearning4j.graph.api.Graph;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.api.VertexSequence;

import java.util.NoSuchElementException;

/**
 * Created by Alex on 9/11/2015.
 */
public class SimpleVertexSequence<V> implements VertexSequence<V> {

    private final Graph<V,?> graph;
    private int[] indices;

    private int currIdx = 0;

    public SimpleVertexSequence( Graph<V,?> graph, int[] indices ){
        this.graph = graph;
        this.indices = indices;
    }

    @Override
    public int sequenceLength() {
        return indices.length;
    }

    public boolean hasNext() {
        return currIdx < indices.length;
    }

    public Vertex<V> next() {
        if(!hasNext()) throw new NoSuchElementException();
        return graph.getVertex(indices[currIdx++]);
    }
}
