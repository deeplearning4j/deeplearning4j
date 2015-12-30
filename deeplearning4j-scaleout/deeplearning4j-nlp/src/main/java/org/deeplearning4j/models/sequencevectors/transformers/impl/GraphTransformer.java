package org.deeplearning4j.models.sequencevectors.transformers.impl;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.Iterator;

/**
 *
 * This class is used to build vocabulary out of graph, via abstract GraphWalkIterator
 *
 * WORK IS IN PROGRESS, DO NOT USE
 * @author raver119@gmail.com
 */
public class GraphTransformer<T extends SequenceElement,V> implements Iterable<T> {
    // TODO: to be implemented

    @Override
    public Iterator<T> iterator() {
        return null;
    }
}
