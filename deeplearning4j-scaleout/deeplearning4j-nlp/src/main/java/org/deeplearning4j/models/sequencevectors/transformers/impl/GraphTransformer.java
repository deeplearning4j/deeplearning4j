package org.deeplearning4j.models.sequencevectors.transformers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkMode;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.graph.walkers.RandomWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.text.labels.LabelsProvider;
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
    protected IGraph sourceGraph;
    protected int walkLength = 5;
    protected WalkMode walkMode = WalkMode.RANDOM;
    protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;
    protected GraphWalker<T> walker;

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
        protected LabelsProvider labelsProvider;

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

        public Builder<T> setLabelsProvider(@NonNull LabelsProvider provider) {
            this.labelsProvider = provider;
            return this;
        }

        public GraphTransformer<T> build() {
            GraphTransformer<T> transformer = new GraphTransformer<T>();
            transformer.noEdgeHandling = this.noEdgeHandling;
            transformer.sourceGraph = this.sourceGraph;
            transformer.walkLength = this.walkLength;
            transformer.walkMode = this.walkMode;

            switch (this.walkMode) {
                case RANDOM:
                    transformer.walker = new RandomWalker.Builder<T>()
                            .setNoEdgeHandling(this.noEdgeHandling)
                            .setWalkLength(this.walkLength)
                            .build();
                    break;
                case WEIGHTED:
                default:
                    throw new UnsupportedOperationException("WalkMode ["+ this.walkMode+"] isn't supported at this moment?");
            }

            return transformer;
        }
    }
}
