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

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.exception.NoEdgesException;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Edge;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.List;
import java.util.Random;

/**
 * This is vertex weight-based walker for SequenceVectors-based DeepWalk implementation.
 * Instead of random walks, this walker produces walks based on weight of the edges.
 *
 * @author AlexDBlack
 * @author raver119@gmail.com
 * Based on Alex Black WeightedWalkIterator implementation
 */
public class WeightedWalker<T extends SequenceElement> extends RandomWalker<T> implements GraphWalker<T> {

    protected WeightedWalker() {

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

    @Override
    public boolean isLabelEnabled() {
        return false;
    }

    /**
     * This method returns next walk sequence from this graph
     *
     * @return
     */
    @Override
    public Sequence<T> next() {
        Sequence<T> sequence = new Sequence<>();

        int startPosition = position.getAndIncrement();
        int lastId = -1;
        int currentPoint = order[startPosition];
        int startPoint = currentPoint;
        for (int i = 0; i < walkLength; i++) {

            if (alpha > 0 && lastId != startPoint && lastId != -1 && alpha > rng.nextDouble()) {
                startPosition = startPoint;
                continue;
            }


            Vertex<T> vertex = sourceGraph.getVertex(currentPoint);
            sequence.addElement(vertex.getValue());

            List<? extends Edge<? extends Number>> edges = sourceGraph.getEdgesOut(currentPoint);

            if (edges == null || edges.isEmpty()) {
                switch (noEdgeHandling) {
                    case CUTOFF_ON_DISCONNECTED:
                        // we just break this sequence
                        i = walkLength;
                        break;
                    case EXCEPTION_ON_DISCONNECTED:
                        throw new NoEdgesException("No available edges left");
                    case PADDING_ON_DISCONNECTED:
                        // TODO: implement padding
                        throw new UnsupportedOperationException("Padding isn't implemented yet");
                    case RESTART_ON_DISCONNECTED:
                        currentPoint = order[startPosition];
                        break;
                    case SELF_LOOP_ON_DISCONNECTED:
                        // we pad walk with this vertex, to do that - we just don't do anything, and currentPoint will be the same till the end of walk
                        break;
                }
            } else {
                double totalWeight = 0.0;
                for (Edge<? extends Number> edge : edges) {
                    totalWeight += edge.getValue().doubleValue();
                }

                double d = rng.nextDouble();
                double threshold = d * totalWeight;
                double sumWeight = 0.0;
                for (Edge<? extends Number> edge : edges) {
                    sumWeight += edge.getValue().doubleValue();
                    if (sumWeight >= threshold) {
                        if (edge.isDirected()) {
                            currentPoint = edge.getTo();
                        } else {
                            if (edge.getFrom() == currentPoint) {
                                currentPoint = edge.getTo();
                            } else {
                                currentPoint = edge.getFrom(); //Undirected edge: might be next--currVertexIdx instead of currVertexIdx--next
                            }
                        }
                        lastId = currentPoint;
                        break;
                    }
                }
            }
        }

        return sequence;
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

        public Builder(IGraph<T, ? extends Number> sourceGraph) {
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
            WeightedWalker<T> walker = new WeightedWalker<>();
            walker.noEdgeHandling = this.noEdgeHandling;
            walker.sourceGraph = this.sourceGraph;
            walker.walkLength = this.walkLength;
            walker.seed = this.seed;
            walker.walkDirection = this.walkDirection;
            walker.alpha = this.alpha;

            walker.order = new int[sourceGraph.numVertices()];
            for (int i = 0; i < walker.order.length; i++) {
                walker.order[i] = i;
            }

            if (this.seed != 0)
                walker.rng = new Random(this.seed);

            return walker;
        }
    }
}
