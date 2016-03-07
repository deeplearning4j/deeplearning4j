package org.deeplearning4j.models.sequencevectors.graph.walkers.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.berkeley.PriorityQueue;
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
                        if (connections.length > 0) {
                            for (int connected : connections) {
                                queue.add(new Node<T>(connected, sourceGraph.getConnectedVertices(connected).size()), sourceGraph.getConnectedVertices(connected).size());
                            }

                            logger.info("Queue: " + queue);
                            Vertex<T> nV =  sourceGraph.getVertex(queue.peek().getVertexId());
                            startPosition = nV.vertexID();
                        } else {
                            i+= 100;
                        }
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown WalkDirection ["+ walkDirection +"]");
            }

        }

        return sequence;
    }

    @Override
    public void reset(boolean shuffle) {
        super.reset(shuffle);
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

    public static class Builder<T extends SequenceElement> extends RandomWalker.Builder<T> {

        public Builder(IGraph<T, ?> sourceGraph) {
            super(sourceGraph);
        }


        @Override
        public GraphWalker<T> build() {
            PopularityWalker<T> walker = new PopularityWalker<T>();
            walker.noEdgeHandling = this.noEdgeHandling;
            walker.sourceGraph = this.sourceGraph;
            walker.walkLength = this.walkLength;
            walker.seed = this.seed;
            walker.walkDirection = this.walkDirection;

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
