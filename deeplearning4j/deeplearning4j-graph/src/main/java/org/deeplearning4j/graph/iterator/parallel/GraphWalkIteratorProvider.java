package org.deeplearning4j.graph.iterator.parallel;

import org.deeplearning4j.graph.iterator.GraphWalkIterator;

import java.util.List;

/**GraphWalkIteratorProvider: implementations of this interface provide a set of GraphWalkIterator objects.
 * Intended use: parallelization. One GraphWalkIterator per thread.
 */
public interface GraphWalkIteratorProvider<V> {

    /**Get a list of GraphWalkIterators. In general: may return less than the specified number of iterators,
     * (for example, for small networks) but never more than it
     * @param numIterators Number of iterators to return
     */
    List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators);

}
