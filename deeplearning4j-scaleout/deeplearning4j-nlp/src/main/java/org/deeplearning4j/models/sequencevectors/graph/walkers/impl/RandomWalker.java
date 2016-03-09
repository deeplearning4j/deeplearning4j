package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

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
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * @author raver119@gmail.com
 *
 * WORK IS IN PROGRESS, DO NOT USE THIS
 *
 * Based on Alex Black RandomWalkIterator implementation
 */
public class RandomWalker<T extends SequenceElement> implements GraphWalker<T> {
    protected int walkLength = 5;
    protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;
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


    @Override
    public boolean hasNext() {
        return position.get() < sourceGraph.numVertices();
    }

    @Override
    public Sequence<T> next() {
        int[] visitedHops = new int[walkLength];
        Arrays.fill(visitedHops, -1);

        Sequence<T> sequence = new Sequence<T>();

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
                    };
                    break;
                case FORWARD_ONLY: {
                        // here we remove only last hop
                        int[] nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition), lastId);
                        if (nextHops.length > 0) {
                            startPosition = nextHops[rng.nextInt(nextHops.length)];
                        } else {
                            switch (noEdgeHandling) {
                                case CUTOFF_ON_DISCONNECTED: {
                                        i += walkLength;
                                    }
                                    break;
                                case EXCEPTION_ON_DISCONNECTED: {
                                        throw new NoEdgesException("No more edges at vertex ["+currentPosition +"]");
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
                                    throw new UnsupportedOperationException("NoEdgeHandling mode ["+noEdgeHandling+"] not implemented yet.");
                            }
                        }
                    };
                    break;
                case FORWARD_UNIQUE: {
                    // here we remove all previously visited hops, and we don't get  back to them ever
                    int[] nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition), visitedHops);
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
                                    throw new NoEdgesException("No more edges at vertex ["+currentPosition +"]");
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
                                throw new UnsupportedOperationException("NoEdgeHandling mode ["+noEdgeHandling+"] not implemented yet.");
                        }
                    }
                };
                break;
                case FORWARD_PREFERRED: {
                        // here we remove all previously visited hops, and if there's no next unique hop available - we fallback to anything, but the last one
                        int[] nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition), visitedHops);
                        if (nextHops.length == 0) {
                            nextHops = ArrayUtils.removeElements(sourceGraph.getConnectedVertexIndices(currentPosition), lastId);
                            if (nextHops.length == 0) {
                                switch (noEdgeHandling) {
                                    case CUTOFF_ON_DISCONNECTED: {
                                            i += walkLength;
                                        }
                                        break;
                                    case EXCEPTION_ON_DISCONNECTED: {
                                            throw new NoEdgesException("No more edges at vertex ["+currentPosition +"]");
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
                                        throw new UnsupportedOperationException("NoEdgeHandling mode ["+noEdgeHandling+"] not implemented yet.");
                                }
                            } else startPosition = nextHops[rng.nextInt(nextHops.length)];
                        }
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown WalkDirection ["+ walkDirection +"]");
            }

            lastId = vertex.vertexID();
        }

        //if (startPoint == 0 || startPoint % 1000 == 0)
            //System.out.println("");
        return sequence;
    }

    @Override
    public void reset(boolean shuffle) {
        this.position.set(0);
        if (shuffle) {
            logger.debug("Calling shuffle() on entries...");
            // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
            for(int i=order.length-1; i>0; i-- ){
                int j = rng.nextInt(i+1);
                int temp = order[j];
                order[j] = order[i];
                order[i] = temp;
            }
        }
    }

    public static class Builder<T extends SequenceElement> {
        protected int walkLength = 5;
        protected NoEdgeHandling noEdgeHandling = NoEdgeHandling.EXCEPTION_ON_DISCONNECTED;
        protected IGraph<T, ?> sourceGraph;
        protected long seed = 0;
        protected WalkDirection walkDirection = WalkDirection.FORWARD_ONLY;
        protected double alpha;

        public Builder(@NonNull IGraph<T, ?> graph) {
            this.sourceGraph = graph;
        }

        public Builder<T> setWalkLength(int walkLength) {
            this.walkLength = walkLength;
            return this;
        }

        public Builder<T> setNoEdgeHandling(@NonNull NoEdgeHandling handling) {
            this.noEdgeHandling = handling;
            return this;
        }

        public Builder<T> setSeed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder<T> setWalkDirection(@NonNull WalkDirection direction) {
            this.walkDirection = direction;
            return this;
        }

        public Builder<T> setRestartProbability(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public RandomWalker<T> build() {
            RandomWalker<T> walker = new RandomWalker<T>();
            walker.noEdgeHandling = this.noEdgeHandling;
            walker.sourceGraph = this.sourceGraph;
            walker.walkLength = this.walkLength;
            walker.seed = this.seed;
            walker.walkDirection = this.walkDirection;
            walker.alpha = this.alpha;

            walker.order = new int[sourceGraph.numVertices()];
            for (int i =0; i <walker.order.length; i++) {
                walker.order[i] = i;
            }

            if (this.seed != 0)
                walker.rng = new Random(this.seed);

            return walker;
        }
    }
}
