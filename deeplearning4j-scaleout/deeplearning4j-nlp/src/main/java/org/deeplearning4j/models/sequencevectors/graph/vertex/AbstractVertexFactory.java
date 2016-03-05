package org.deeplearning4j.models.sequencevectors.graph.vertex;

import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public class AbstractVertexFactory<T extends SequenceElement> implements VertexFactory<T> {

    @Override
    public Vertex<T> create(int vertexIdx) {
        return null;
    }

    @Override
    public Vertex<T> create(T element) {
        return null;
    }

    @Override
    public Vertex<T> create(int vertexIdx, T element) {
        return null;
    }
}
