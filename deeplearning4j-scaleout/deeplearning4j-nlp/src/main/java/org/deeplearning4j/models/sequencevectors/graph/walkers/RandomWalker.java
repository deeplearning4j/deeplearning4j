package org.deeplearning4j.models.sequencevectors.graph.walkers;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public class RandomWalker<T extends SequenceElement> implements GraphWalker<T> {

    protected RandomWalker() {

    }

    public RandomWalker(int walkLength, NoEdgeHandling noEdgeHandling) {

    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public Sequence<T> next() {
        return null;
    }

    @Override
    public void reset() {

    }

    public static class Builder<T extends SequenceElement> {
        protected int walkLength = 5;
        protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;

        public Builder() {
            ;
        }

        public Builder<T> setWalkLength(int walkLength) {
            this.walkLength = walkLength;
            return this;
        }

        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            this.noEdgeHandling = handling;
            return this;
        }

        public RandomWalker<T> build() {
            RandomWalker walker = new RandomWalker(this.walkLength, this.noEdgeHandling);

            return walker;
        }
    }
}
