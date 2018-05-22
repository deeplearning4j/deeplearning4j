package org.deeplearning4j.models.sequencevectors.graph.walkers;

import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This interface describes methods needed for various DeepWalk-related implementations
 *
 * @author raver119@gmail.com
 */
public interface GraphWalker<T extends SequenceElement> {

    IGraph<T, ?> getSourceGraph();

    /**
     * This method checks, if walker has any more sequences left in queue
     *
     * @return
     */
    boolean hasNext();

    /**
     * This method returns next walk sequence from this graph
     *
     * @return
     */
    Sequence<T> next();

    /**
     * This method resets walker
     *
     * @param shuffle if TRUE, order of walks will be shuffled
     */
    void reset(boolean shuffle);


    boolean isLabelEnabled();
}
