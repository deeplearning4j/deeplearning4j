package org.deeplearning4j.graph.graph;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.IVertexSequence;
import org.deeplearning4j.graph.api.Vertex;

import java.util.NoSuchElementException;

/**A vertex sequence represents a sequences of vertices in a graph
 * @author Alex Black
 */
public class VertexSequence<V> implements IVertexSequence<V> {
    private final IGraph<V, ?> graph;
    private int[] indices;
    private int currIdx = 0;

    public VertexSequence(IGraph<V, ?> graph, int[] indices) {
        this.graph = graph;
        this.indices = indices;
    }

    @Override
    public int sequenceLength() {
        return indices.length;
    }

    @Override
    public boolean hasNext() {
        return currIdx < indices.length;
    }

    @Override
    public Vertex<V> next() {
        if (!hasNext())
            throw new NoSuchElementException();
        return graph.getVertex(indices[currIdx++]);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
