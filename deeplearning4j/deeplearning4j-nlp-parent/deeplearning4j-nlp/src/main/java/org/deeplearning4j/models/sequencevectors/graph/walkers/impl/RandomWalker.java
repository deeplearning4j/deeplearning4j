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

import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.exception.NoEdgesException;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.graph.walkers.GraphWalker;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is Random-based walker for SequenceVectors-based DeepWalk implementation
 *
 * Original DeepWalk paper: <a href="http://arxiv.org/pdf/1403.6652v2">http://arxiv.org/pdf/1403.6652v2</a>
 *
 * @author AlexDBlack
 * @author raver119@gmail.com
 *
 * Based on Alex Black RandomWalkIterator implementation
 */
public class RandomWalker<T extends SequenceElement> implements GraphWalker<T> {
    protected int walkLength = 5;
    protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;
    @Getter
    protected IGraph<T, ?> sourceGraph;
    protected AtomicInteger position = new AtomicInteger(0);
    protected Random rng = new Random(System.currentTimeMillis());
    protected long seed;
    protected int[] order;
    protected WalkDirection walkDirection;
    protected double alpha;

    private static final Logger logger = LoggerFactory.getLogger(RandomWalker.class);

    protected RandomWalker() {

    }


    /**
     * This method checks, if walker has any more sequences left in queue
     *
     * @return
     */
    @Override
    public boolean hasNext() {
        return position.get() < sourceGraph.numVertices();
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
        int[] visitedHops = new int[walkLength];
        Arrays.fill(visitedHops, -1);

        Sequence<T> sequence = new Sequence<>();

        int startPosition = position.getAndIncrement();
        int lastId = -1;
        int startPoint = order[startPosition];
        //System.out.println("");


        startPosition = startPoint;

        //if (startPosition == 0 || startPoint % 1000 == 0)
        //   System.out.println("ATZ Walk: ");

        for (int i = 0; i < walkLength; i++) {
            Vertex<T> vertex = sourceGraph.getVertex(startPosition);

            int currentPosition = startPosition;

            sequence.addElement(vertex.getValue());
            visitedHops[i] = vertex.vertexID();
            //if (startPoint == 0 || startPoint % 1000 == 0)
            // System.out.print("" + vertex.vertexID() + " -> ");


            if (alpha > 0 && lastId != startPoint && lastId != -1 && alpha > rng.nextDouble()) {
                startPosition = startPoint;
                continue;
            }


            // get next vertex
            switch (walkDirection) {
                case RANDOM: {
                    int[] nextHops = sourceGraph.getConnectedVertexIndices(currentPosition);
                    startPosition = nextHops[rng.nextInt(nextHops.length)];
                }
                    break;
                case FORWARD_ONLY: {
                    // here we remove only last hop
                    int[] nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition),
                                    lastId);
                    if (nextHops.length > 0) {
                        startPosition = nextHops[rng.nextInt(nextHops.length)];
                    } else {
                        switch (noEdgeHandling) {
                            case CUTOFF_ON_DISCONNECTED: {
                                i += walkLength;
                            }
                                break;
                            case EXCEPTION_ON_DISCONNECTED: {
                                throw new NoEdgesException("No more edges at vertex [" + currentPosition + "]");
                            }
                            case SELF_LOOP_ON_DISCONNECTED: {
                                startPosition = currentPosition;
                            }
                                break;
                            case PADDING_ON_DISCONNECTED: {
                                throw new UnsupportedOperationException("PADDING not implemented yet");
                            }
                            case RESTART_ON_DISCONNECTED: {
                                startPosition = startPoint;
                            }
                                break;
                            default:
                                throw new UnsupportedOperationException(
                                                "NoEdgeHandling mode [" + noEdgeHandling + "] not implemented yet.");
                        }
                    }
                }
                    break;
                case FORWARD_UNIQUE: {
                    // here we remove all previously visited hops, and we don't get  back to them ever
                    int[] nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition),
                                    visitedHops);
                    if (nextHops.length > 0) {
                        startPosition = nextHops[rng.nextInt(nextHops.length)];
                    } else {
                        // if we don't have any more unique hops within this path - break out.
                        switch (noEdgeHandling) {
                            case CUTOFF_ON_DISCONNECTED: {
                                i += walkLength;
                            }
                                break;
                            case EXCEPTION_ON_DISCONNECTED: {
                                throw new NoEdgesException("No more edges at vertex [" + currentPosition + "]");
                            }
                            case SELF_LOOP_ON_DISCONNECTED: {
                                startPosition = currentPosition;
                            }
                                break;
                            case PADDING_ON_DISCONNECTED: {
                                throw new UnsupportedOperationException("PADDING not implemented yet");
                            }
                            case RESTART_ON_DISCONNECTED: {
                                startPosition = startPoint;
                            }
                                break;
                            default:
                                throw new UnsupportedOperationException(
                                                "NoEdgeHandling mode [" + noEdgeHandling + "] not implemented yet.");
                        }
                    }
                }
                    break;
                case FORWARD_PREFERRED: {
                    // here we remove all previously visited hops, and if there's no next unique hop available - we fallback to anything, but the last one
                    int[] nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition),
                                    visitedHops);
                    if (nextHops.length == 0) {
                        nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition),
                                        lastId);
                        if (nextHops.length == 0) {
                            switch (noEdgeHandling) {
                                case CUTOFF_ON_DISCONNECTED: {
                                    i += walkLength;
                                }
                                    break;
                                case EXCEPTION_ON_DISCONNECTED: {
                                    throw new NoEdgesException("No more edges at vertex [" + currentPosition + "]");
                                }
                                case SELF_LOOP_ON_DISCONNECTED: {
                                    startPosition = currentPosition;
                                }
                                    break;
                                case PADDING_ON_DISCONNECTED: {
                                    throw new UnsupportedOperationException("PADDING not implemented yet");
                                }
                                case RESTART_ON_DISCONNECTED: {
                                    startPosition = startPoint;
                                }
                                    break;
                                default:
                                    throw new UnsupportedOperationException("NoEdgeHandling mode [" + noEdgeHandling
                                                    + "] not implemented yet.");
                            }
                        } else
                            startPosition = nextHops[rng.nextInt(nextHops.length)];
                    }
                }
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown WalkDirection [" + walkDirection + "]");
            }

            lastId = vertex.vertexID();
        }

        //if (startPoint == 0 || startPoint % 1000 == 0)
        //System.out.println("");
        return sequence;
    }

    /**
     * This method resets walker
     *
     * @param shuffle if TRUE, order of walks will be shuffled
     */
    @Override
    public void reset(boolean shuffle) {
        this.position.set(0);
        if (shuffle) {
            logger.debug("Calling shuffle() on entries...");
            // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
            for (int i = order.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int temp = order[j];
                order[j] = order[i];
                order[i] = temp;
            }
        }
    }

    public static class Builder<T extends SequenceElement> {
        protected int walkLength = 5;
        protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.RESTART_ON_DISCONNECTED;
        protected IGraph<T, ?> sourceGraph;
        protected long seed = 0;
        protected WalkDirection walkDirection = WalkDirection.FORWARD_ONLY;
        protected double alpha;

        /**
         * Builder constructor for RandomWalker
         *
         * @param graph source graph to be used for this walker
         */
        public Builder(@NonNull IGraph<T, ?> graph) {
            this.sourceGraph = graph;
        }

        /**
         * This method specifies output sequence (walk) length
         *
         * @param walkLength
         * @return
         */
        public Builder<T> setWalkLength(int walkLength) {
            this.walkLength = walkLength;
            return this;
        }

        /**
         * This method defines walker behavior when it gets to node which has no next nodes available
         * Default value: RESTART_ON_DISCONNECTED
         *
         * @param handling
         * @return
         */
        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            this.noEdgeHandling = handling;
            return this;
        }

        /**
         * This method specifies random seed.
         *
         * @param seed
         * @return
         */
        public Builder<T> setSeed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * This method defines next hop selection within walk
         *
         * @param direction
         * @return
         */
        public Builder<T> setWalkDirection(@NonNull WalkDirection direction) {
            this.walkDirection = direction;
            return this;
        }

        /**
         * This method defines a chance for walk restart
         * Good value would be somewhere between 0.03-0.07
         *
         * @param alpha
         * @return
         */
        public Builder<T> setRestartProbability(double alpha) {
            this.alpha = alpha;
            return this;
        }

        /**
         * This method builds RandomWalker instance
         * @return
         */
        public RandomWalker<T> build() {
            RandomWalker<T> walker = new RandomWalker<>();
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
