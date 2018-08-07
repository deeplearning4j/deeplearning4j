/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.math3.util.MathArrays;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * This is vertex popularity-based walker for SequenceVectors-based DeepWalk implementation.
 * Instead of rand walks, this walker produces walks based on number of edges coming into each node.
 * This allows you to build walks filtering too rare nodes, or too popular nodes, depending on your demands.
 *
 * Original DeepWalk paper: http://arxiv.org/pdf/1403.6652v2
 * @author raver119@gmail.com
 */
public class PopularityWalker<T extends SequenceElement> extends RandomWalker<T> implements GraphWalker<T> {
    protected PopularityMode popularityMode = PopularityMode.MAXIMUM;
    protected int spread = 10;
    protected SpreadSpectrum spectrum;

    private static final Logger logger = LoggerFactory.getLogger(PopularityWalker.class);

    /**
     * This method checks, if walker has any more sequences left in queue
     *
     * @return
     */
    @Override
    public boolean hasNext() {
        return super.hasNext();
    }

    @Override
    public boolean isLabelEnabled() {
        return false;
    }


    protected class NodeComparator implements Comparator<Node<T>> {

        @Override
        public int compare(Node<T> o1, Node<T> o2) {
            return Integer.compare(o2.weight, o1.weight);
        }
    }

    /**
     * This method returns next walk sequence from this graph
     *
     * @return
     */
    @Override
    public Sequence<T> next() {
        Sequence<T> sequence = new Sequence<>();
        int[] visitedHops = new int[walkLength];
        Arrays.fill(visitedHops, -1);

        int startPosition = position.getAndIncrement();
        int lastId = -1;
        int startPoint = order[startPosition];
        startPosition = startPoint;
        for (int i = 0; i < walkLength; i++) {

            Vertex<T> vertex = sourceGraph.getVertex(startPosition);

            int currentPosition = startPosition;

            sequence.addElement(vertex.getValue());
            visitedHops[i] = vertex.vertexID();
            int cSpread = 0;

            if (alpha > 0 && lastId != startPoint && lastId != -1 && alpha > rng.nextDouble()) {
                startPosition = startPoint;
                continue;
            }

            switch (walkDirection) {
                case RANDOM:
                case FORWARD_ONLY:
                case FORWARD_UNIQUE:
                case FORWARD_PREFERRED: {

                    // ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(order[currentPosition]), visitedHops);
                    int[] connections = ArrayUtils.removeElements(
                                    sourceGraph.getConnectedVertexIndices(vertex.vertexID()), visitedHops);

                    // we get  popularity of each node connected to the current node.
                    PriorityQueue<Node<T>> queue = new PriorityQueue<>(Math.max(10, connections.length), new NodeComparator());

                    int start = 0;
                    int stop = 0;
                    int cnt = 0;
                    if (connections.length > 0) {


                        for (int connected : connections) {
                            Node<T> tNode = new Node<>(connected, sourceGraph.getConnectedVertices(connected).size());
                            queue.add(tNode);
                        }


                        cSpread = spread > connections.length ? connections.length : spread;

                        switch (popularityMode) {
                            case MAXIMUM:
                                start = 0;
                                stop = start + cSpread - 1;
                                break;
                            case MINIMUM:
                                start = connections.length - cSpread;
                                stop = connections.length - 1;
                                break;
                            case AVERAGE:
                                int mid = connections.length / 2;
                                start = mid - (cSpread / 2);
                                stop = mid + (cSpread / 2);
                                break;
                        }

                        // logger.info("Spread: ["+ cSpread+ "], Connections: ["+ connections.length+"], Start: ["+start+"], Stop: ["+stop+"]");
                        cnt = 0;
                        //logger.info("Queue: " + queue);
                        //logger.info("Queue size: " + queue.size());

                        List<Node<T>> list = new ArrayList<>();
                        double[] weights = new double[cSpread];

                        int fcnt = 0;
                        while (!queue.isEmpty()) {
                            Node<T> node = queue.poll();
                            if (cnt >= start && cnt <= stop) {
                                list.add(node);
                                weights[fcnt] = node.getWeight();
                                fcnt++;
                            }
                            connections[cnt] = node.getVertexId();

                            cnt++;
                        }


                        int con = -1;

                        switch (spectrum) {
                            case PLAIN: {
                                con = RandomUtils.nextInt(start, stop + 1);

                                //    logger.info("Picked selection: " + con);

                                Vertex<T> nV = sourceGraph.getVertex(connections[con]);
                                startPosition = nV.vertexID();
                                lastId = vertex.vertexID();
                            }
                                break;
                            case PROPORTIONAL: {
                                double norm[] = MathArrays.normalizeArray(weights, 1);
                                double prob = rng.nextDouble();
                                double floor = 0.0;
                                for (int b = 0; b < weights.length; b++) {
                                    if (prob >= floor && prob < floor + norm[b]) {
                                        startPosition = list.get(b).getVertexId();
                                        lastId = startPosition;
                                        break;
                                    } else {
                                        floor += norm[b];
                                    }
                                }
                            }
                                break;
                        }

                    } else {
                        switch (noEdgeHandling) {
                            case EXCEPTION_ON_DISCONNECTED:
                                throw new NoEdgesException("No more edges at vertex [" + currentPosition + "]");
                            case CUTOFF_ON_DISCONNECTED:
                                i += walkLength;
                                break;
                            case SELF_LOOP_ON_DISCONNECTED:
                                startPosition = currentPosition;
                                break;
                            case RESTART_ON_DISCONNECTED:
                                startPosition = startPoint;
                                break;
                            default:
                                throw new UnsupportedOperationException(
                                                "Unsupported noEdgeHandling: [" + noEdgeHandling + "]");
                        }
                    }
                }
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown WalkDirection: [" + walkDirection + "]");
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


        /**
         * This method defines which nodes should be taken in account when choosing next hope: maximum popularity, lowest popularity, or average popularity.
         * Default value: MAXIMUM
         *
         * @param popularityMode
         * @return
         */
        public Builder<T> setPopularityMode(@NonNull PopularityMode popularityMode) {
            this.popularityMode = popularityMode;
            return this;
        }

        /**
         * This method defines, how much nodes should take place in next hop selection. Something like top-N nodes, or bottom-N nodes.
         * Default value: 10
         *
         * @param topN
         * @return
         */
        public Builder<T> setPopularitySpread(int topN) {
            this.spread = topN;
            return this;
        }

        /**
         * This method allows you to define, if nodes within popularity spread should have equal chances to be picked for next hop, or they should have chances proportional to their popularity.
         *
         * Default value: PLAIN
         *
         * @param spectrum
         * @return
         */
        public Builder<T> setSpreadSpectrum(@NonNull SpreadSpectrum spectrum) {
            this.spectrum = spectrum;
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
         * This method defines a chance for walk restart
         * Good value would be somewhere between 0.03-0.07
         *
         * @param alpha
         * @return
         */
        @Override
        public Builder<T> setRestartProbability(double alpha) {
            super.setRestartProbability(alpha);
            return this;
        }

        /**
         * This method builds PopularityWalker object with previously specified params
         *
         * @return
         */
        @Override
        public PopularityWalker<T> build() {
            PopularityWalker<T> walker = new PopularityWalker<>();
            walker.noEdgeHandling = this.noEdgeHandling;
            walker.sourceGraph = this.sourceGraph;
            walker.walkLength = this.walkLength;
            walker.seed = this.seed;
            walker.walkDirection = this.walkDirection;
            walker.alpha = this.alpha;
            walker.popularityMode = this.popularityMode;
            walker.spread = this.spread;
            walker.spectrum = this.spectrum;

            walker.order = new int[sourceGraph.numVertices()];
            for (int i = 0; i < walker.order.length; i++) {
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
