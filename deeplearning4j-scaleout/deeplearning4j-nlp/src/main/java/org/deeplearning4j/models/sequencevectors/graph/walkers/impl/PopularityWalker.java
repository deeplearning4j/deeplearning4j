package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.berkeley.PriorityQueue;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.PopularityMode;
import org.deeplearning4j.models.sequencevectors.graph.enums.SpreadSpectrum;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.exception.NoEdgesException;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author raver119@gmail.com
 */
public class PopularityWalker<T extends SequenceElement> extends RandomWalker<T>  implements GraphWalker<T> {
    protected PopularityMode popularityMode = PopularityMode.MAXIMUM;
    protected int spread = 10;
    protected SpreadSpectrum spectrum;

    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    public Sequence<T> next() {
        Sequence<T> sequence = new Sequence<>();
        int[] visitedHops = new int[walkLength];
        Arrays.fill(visitedHops, -1);

        int startPosition = position.getAndIncrement();
        int lastId = -1;
        int startPoint = order[startPosition];
        for (int i = 0; i < walkLength; i++) {
            int currentPosition = startPosition;
            Vertex<T> vertex = sourceGraph.getVertex(order[currentPosition]);
            sequence.addElement(vertex.getValue());
            visitedHops[i] = vertex.vertexID();

            switch (walkDirection) {
                case RANDOM:
                case FORWARD_ONLY:
                case FORWARD_PREFERRED: {
                        // we get  popularity of each node connected to the current node.
                        PriorityQueue<Node<T>> queue = new PriorityQueue<Node<T>>();

                        // ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(order[currentPosition]), visitedHops);
                        int[] connections = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(vertex.vertexID()), visitedHops);
                        int start = 0;
                        int stop = 0;
                        if (connections.length > 0) {

                            for (int connected : connections) {
                                queue.add(new Node<T>(connected, sourceGraph.getConnectedVertices(connected).size()), sourceGraph.getConnectedVertices(connected).size());
                            }

                            spread = spread > connections.length ? connections.length : spread;

                            switch (popularityMode) {
                                case MAXIMUM:
                                    start = 0;
                                    stop = start + spread - 1;
                                    break;
                                case MINIMUM:
                                    start = connections.length - spread;
                                    stop = connections.length - 1;
                                    break;
                                case AVERAGE:
                                    int mid = connections.length / 2;
                                    start = mid - (spread/2);
                                    stop = mid + (spread / 2);
                                    break;
                            }

                            logger.info("Spread: ["+ spread+ "], Connections: ["+ connections.length+"], Start: ["+start+"], Stop: ["+stop+"]");
                            int cnt = 0;
                            logger.info("Queue: " + queue);
                            logger.info("Queue size: " + queue.size());

                            while (queue.hasNext()) {
                                connections[cnt] = queue.next().getVertexId();
                                cnt++;
                            }


                            int con = 0;

                            switch (spectrum) {
                                case PLAIN:
                                        con = RandomUtils.nextInt(start, stop + 1);
                                    break;
                                case PROPORTIONAL:

                                    break;
                            }


                            logger.info("Picked selection: " + con);

                            Vertex<T> nV =  sourceGraph.getVertex(connections[con]);
                            startPosition = nV.vertexID();
                        } else {
                            switch (noEdgeHandling) {
                                case EXCEPTION_ON_DISCONNECTED:
                                    throw new NoEdgesException("No more edges at vertex ["+currentPosition +"]");
                                case CUTOFF_ON_DISCONNECTED:
                                    i += walkLength;
                                    break;
                                case SELF_LOOP_ON_DISCONNECTED:
                                    startPosition = currentPosition;
                                    break;
                                default:
                                    throw new UnsupportedOperationException("Unsupported noEdgeHandling: ["+ noEdgeHandling+"]");
                            }
                        }
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown WalkDirection: ["+ walkDirection +"]");
            }

        }

        return sequence;
    }

    @Override
    public void reset(boolean shuffle) {
        super.reset(shuffle);
    }

    public static class Builder<T extends SequenceElement> extends RandomWalker.Builder<T> {
        protected PopularityMode popularityMode = PopularityMode.MAXIMUM;
        protected int spread = 10;
        protected SpreadSpectrum spectrum = SpreadSpectrum.PLAIN;

        public Builder(IGraph<T, ?> sourceGraph) {
            super(sourceGraph);
        }


        public Builder<T> setPopularityMode(@NonNull PopularityMode popularityMode) {
            this.popularityMode = popularityMode;
            return this;
        }

        public Builder<T> setPopularitySpread(int topN) {
            this.spread = topN;
            return this;
        }

        public Builder<T> setSpreadSpectrum(@NonNull SpreadSpectrum spectrum) {
            this.spectrum = spectrum;
            return this;
        }

        @Override
        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            super.setNoEdgeHandling(handling);
            return this;
        }

        @Override
        public Builder<T> setSeed(long seed) {
            super.setSeed(seed);
            return this;
        }

        @Override
        public Builder<T> setWalkDirection(@NonNull WalkDirection direction) {
            super.setWalkDirection(direction);
            return this;
        }

        @Override
        public Builder<T> setWalkLength(int walkLength) {
            super.setWalkLength(walkLength);
            return this;
        }

        @Override
        public GraphWalker<T> build() {
            PopularityWalker<T> walker = new PopularityWalker<T>();
            walker.noEdgeHandling = this.noEdgeHandling;
            walker.sourceGraph = this.sourceGraph;
            walker.walkLength = this.walkLength;
            walker.seed = this.seed;
            walker.walkDirection = this.walkDirection;
            walker.popularityMode = this.popularityMode;
            walker.spread = this.spread;
            walker.spectrum = this.spectrum;

            walker.order = new int[sourceGraph.numVertices()];
            for (int i =0; i <walker.order.length; i++) {
                walker.order[i] = i;
            }

            if (this.seed != 0)
                walker.rng = new Random(this.seed);

            return walker;
        }
    }

    @AllArgsConstructor
    @Data
    private static class Node<T extends SequenceElement> implements Comparable<Node<T>> {
        private int vertexId;
        private int weight = 0;

        @Override
        public int compareTo(Node<T> o) {
            return Integer.compare(this.weight, o.weight);
        }
    }
}
