package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This is vertex eight-based walker for SequenceVectors-based DeepWalk implementation.
 * Instead of random walks, this walker produces walks based on weight of the vertices.
 *
 * @author AlexDBlack
 * @author raver119@gmail.com
 * Based on Alex Black WeightedWalkIterator implementation
 */
public class WeightedWalker<T extends SequenceElement> extends RandomWalker<T>  implements GraphWalker<T> {

    protected WeightedWalker(IGraph<T, ?> sourceGraph) {

    }

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

    public static class Builder<T extends SequenceElement> extends RandomWalker.Builder<T>  {

        public Builder(IGraph<T, ?> sourceGraph) {
            super(sourceGraph);
        }

        /**
         * This method specifies output sequence (walk) length
         *
         * @param walkLength
         * @return
         */
        @Override
        public Builder<T> setWalkLength(int walkLength) {
            super.setWalkLength(walkLength);
            return this;
        }

        /**
         * This method defines walker behavior when it gets to node which has no next nodes available
         * Default value: RESTART_ON_DISCONNECTED
         *
         * @param handling
         * @return
         */
        @Override
        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            super.setNoEdgeHandling(handling);
            return this;
        }

        /**
         * This method specifies random seed.
         *
         * @param seed
         * @return
         */
        @Override
        public Builder<T> setSeed(long seed) {
            super.setSeed(seed);
            return this;
        }

        /**
         * This method defines next hop selection within walk
         *
         * @param direction
         * @return
         */
        @Override
        public Builder<T> setWalkDirection(@NonNull WalkDirection direction) {
            super.setWalkDirection(direction);
            return this;
        }

        /**
         * This method defines a chance for walk restart
         * Good value would be somewhere between 0.03-0.07
         *
         * @param alpha
         * @return
         */
        @Override
        public RandomWalker.Builder<T> setRestartProbability(double alpha) {
            return super.setRestartProbability(alpha);
        }

        public WeightedWalker<T> build() {
            WeightedWalker<T> walker = new WeightedWalker<>(sourceGraph);

            return walker;
        }
    }
}
