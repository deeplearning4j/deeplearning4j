package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public class WeightedWalker<T extends SequenceElement> extends RandomWalker<T>  implements GraphWalker<T> {
    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    public Sequence<T> next() {
        return null;
    }

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
