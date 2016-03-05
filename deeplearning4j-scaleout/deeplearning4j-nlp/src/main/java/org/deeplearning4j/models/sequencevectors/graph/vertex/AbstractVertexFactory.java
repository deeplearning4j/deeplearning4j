package org.deeplearning4j.models.sequencevectors.graph.vertex;

import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public class AbstractVertexFactory<T extends SequenceElement> implements VertexFactory<T> {

    @Override
    public Vertex<T> create(int vertexIdx) {
        Vertex<T> vertex = new Vertex<>(vertexIdx, null);
        return vertex;
    }

    @Override
    public Vertex<T> create(int vertexIdx, T element) {
        Vertex<T> vertex = new Vertex<>(vertexIdx, element);
        return vertex;
    }
}
