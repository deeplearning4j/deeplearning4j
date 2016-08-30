package org.deeplearning4j.graph.iterator;

import org.deeplearning4j.graph.api.IVertexSequence;

/**Interface/iterator representing a sequence of walks on a graph
 * For example, a {@code GraphWalkIterator<T>} can represesnt a set of independent random walks on a graph
 */
public interface GraphWalkIterator<T> {

    /** Length of the walks returned by next()
     * Note that a walk of length {@code i} contains {@code i+1} vertices
     */
    int walkLength();

    /**Get the next vertex sequence.
     */
    IVertexSequence<T> next();

    /** Whether the iterator has any more vertex sequences. */
    boolean hasNext();

    /** Reset the graph walk iterator. */
    void reset();
}
