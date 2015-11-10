package org.deeplearning4j.graph.iterator;

import org.deeplearning4j.graph.api.VertexSequence;

/**
 * Created by Alex on 9/11/2015.
 */
public interface GraphWalkIterator<T> {

    public int walkLength();

    VertexSequence<T> next();

    boolean hasNext();

    void reset();



}
