package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;


public interface EdgeLineProcessor<E> {

    /** Process a line of text into an edge.
     * May return null if line is not a valid edge (i.e., comment line etc)
     */
    Edge<E> processLine(String line);

}
