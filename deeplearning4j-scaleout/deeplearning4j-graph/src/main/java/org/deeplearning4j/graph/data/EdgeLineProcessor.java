package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;

/** EdgeLineProcessor is used during data loading from a file, where each edge is on a separate line<br>
 * Provides flexibility in loading graphs with arbitrary objects/properties that can be represented in a text format
 * Can also be used handle conversion of edges between non-numeric vertices to an appropriate numbered format
 * @param <E> type of the edge returned
 */
public interface EdgeLineProcessor<E> {

    /** Process a line of text into an edge.
     * May return null if line is not a valid edge (i.e., comment line etc)
     */
    Edge<E> processLine(String line);

}
