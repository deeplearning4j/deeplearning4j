package org.deeplearning4j.graph.iterator.parallel;

import org.deeplearning4j.graph.iterator.GraphWalkIterator;

import java.util.List;

/**
 */
public interface GraphWalkIteratorProvider<V> {

    List<GraphWalkIterator<V>> getGraphWalkIterators( int numIterators );

}
