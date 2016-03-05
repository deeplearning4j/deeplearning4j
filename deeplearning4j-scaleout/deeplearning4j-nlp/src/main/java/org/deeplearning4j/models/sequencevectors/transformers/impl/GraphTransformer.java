package org.deeplearning4j.models.sequencevectors.transformers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.graph.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkMode;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import scala.collection.Seq;

import java.util.Iterator;

/**
 *
 * This class is used to build vocabulary out of graph, via abstract GraphWalkIterator
 *
 * WORK IS IN PROGRESS, DO NOT USE
 * @author raver119@gmail.com
 */
public class GraphTransformer<T extends SequenceElement> implements Iterable<Sequence<T>> {
    // TODO: to be implemented

    protected GraphTransformer() {
        ;
    }

    @Override
    public Iterator<Sequence<T>> iterator() {
        return null;
    }

    public static class Builder<T extends SequenceElement> {
        protected IGraph sourceGraph;
        protected int walkLength = 5;
        protected WalkMode walkMode = WalkMode.RANDOM;
        protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;

        public Builder(IGraph<T, ?> sourceGraph) {
            this.sourceGraph = sourceGraph;
        }

        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            this.noEdgeHandling = handling;
            return this;
        }

        public Builder<T> setWalkMode(@NonNull WalkMode walkMode) {
            this.walkMode = walkMode;
            return this;
        }

        public Builder<T> setWalkLength(int walkLength) {
            this.walkLength = walkLength;
            return this;
        }

        public GraphTransformer<T> build() {
            GraphTransformer<T> transformer = new GraphTransformer<T>();

            return transformer;
        }
    }
}
