package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.graph.enums.PopularityMode;
import org.deeplearning4j.models.sequencevectors.graph.enums.SamplingMode;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This walker represents connections of a given node.
 * Basically it's the same idea as context for a given node.
 *
 * So this walker produces Sequences, with label defined. And label - is element itself.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class NearestVertexWalker<V extends SequenceElement> implements GraphWalker<V> {
    @Getter protected IGraph<V, ?> sourceGraph;
    protected int walkLength = 0;
    protected long seed = 0;
    protected SamplingMode samplingMode = SamplingMode.RANDOM;
    protected int[] order;
    protected Random rng;

    private AtomicInteger position = new AtomicInteger(0);

    protected NearestVertexWalker() {

    }

    @Override
    public boolean hasNext() {
        return position.get() < order.length;
    }

    @Override
    public Sequence<V> next() {
        return walk(sourceGraph.getVertex(order[position.getAndIncrement()]));
    }

    @Override
    public void reset(boolean shuffle) {
        position.set(0);
        if (shuffle) {
            log.debug("Calling shuffle() on entries...");
            // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
            for(int i=order.length-1; i>0; i-- ){
                int j = rng.nextInt(i+1);
                int temp = order[j];
                order[j] = order[i];
                order[i] = temp;
            }
        }
    }

    protected Sequence<V> walk(Vertex<V> node) {
        Sequence<V> sequence = new Sequence<>();

        int idx = node.vertexID();
        List<Vertex<V>> vertices = sourceGraph.getConnectedVertices(idx);

        sequence.setSequenceLabel(node.getValue());

        if (walkLength == 0) {
            // if walk is unlimited - we use all connected vertices as is
            for (Vertex<V> vertex: vertices)
                sequence.addElement(vertex.getValue());
        } else {
            // if walks are limited, we care about sampling mode
            switch (samplingMode) {
                case MAX_POPULARITY:
                case MEDIAN_POPULARITY:
                case MIN_POPULARITY: {
                    throw new UnsupportedOperationException();
                }
                case RANDOM: {
                    // we randomly sample some number of connected vertices
                    if (vertices.size() <= walkLength)
                        for (Vertex<V> vertex: vertices)
                            sequence.addElement(vertex.getValue());
                    else {
                        Set<V> elements = new HashSet<>();
                        while (elements.size() < walkLength) {
                            Vertex<V> vertex = ArrayUtil.getRandomElement(vertices);
                            elements.add(vertex.getValue());
                        }
                        sequence.addElements(elements);
                    }
                }
                break;
                default:
                    throw new ND4JIllegalStateException("Unknown sampling mode was passed in: [" + samplingMode + "]");
            }
        }

        return sequence;
    }

    @Override
    public boolean isLabelEnabled() {
        return true;
    }

    public static class Builder<V extends SequenceElement> {
        protected int walkLength = 0;
        protected IGraph<V, ?> sourceGraph;
        protected SamplingMode samplingMode = SamplingMode.RANDOM;
        protected long seed;

        public Builder(@NonNull IGraph<V, ?> graph) {
            this.sourceGraph = graph;
        }

        public Builder setSeed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * This method defines maximal number of nodes to be visited during walk.
         *
         * PLEASE NOTE: If set to 0 - no limits will be used.
         *
         * @param length
         * @return
         */
        public Builder setWalkLength(int length){
            walkLength = length;
            return this;
        }

        /**
         * This method defines sorting which will be used to generate walks.
         *
         * PLEASE NOTE: This option has effect only if walkLength is limited (>0).
         *
         * @param mode
         * @return
         */
        public Builder setSamplingMode(@NonNull SamplingMode mode) {
            this.samplingMode = mode;
            return this;
        }

        /**
         * This method returns you new GraphWalker instance
         *
         * @return
         */
        public NearestVertexWalker<V> build() {
            NearestVertexWalker<V> walker = new NearestVertexWalker<>();
            walker.sourceGraph = this.sourceGraph;
            walker.walkLength = this.walkLength;
            walker.samplingMode = this.samplingMode;

            walker.order = new int[sourceGraph.numVertices()];
            for (int i =0; i < walker.order.length; i++) {
                walker.order[i] = i;
            }

            walker.rng = new Random(seed);

            walker.reset(true);

            return walker;
        }
    }
}
