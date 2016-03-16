package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 *
 * WORK IS IN PROGRESS, DO NOT USE THIS
 *
 * @author raver119@gmail.com
 */
public class WeightedWalker<T extends SequenceElement> extends RandomWalker<T>  implements GraphWalker<T> {

    /**
     * This method checks, if walker has any more sequences left in queue
     *
     * @return
     */
    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    /**
     * This method returns next walk sequence from this graph
     *
     * @return
     */
    @Override
    public Sequence<T> next() {
        return null;
    }

    /**
     * This method resets walker
     *
     * @param shuffle if TRUE, order of walks will be shuffled
     */
    @Override
    public void reset(boolean shuffle) {
        super.reset(shuffle);
    }

    public static class Builder<T extends SequenceElement> extends RandomWalker.Builder<T> {

        public Builder(IGraph<T, ?> sourceGraph) {
            super(sourceGraph);
        }

    }
}
