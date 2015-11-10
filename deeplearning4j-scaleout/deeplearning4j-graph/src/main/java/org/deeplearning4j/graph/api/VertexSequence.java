package org.deeplearning4j.graph.api;

import java.util.Iterator;

/**
 * Created by Alex on 9/11/2015.
 */
public interface VertexSequence<T> extends Iterator<Vertex<T>> {

    public int sequenceLength();

}
