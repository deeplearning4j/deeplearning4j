package org.deeplearning4j.graph.iterator.parallel;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.WeightedRandomWalkIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**Weighted random walk graph iterator provider: given a weighted graph (of type {@code IGraph<?,? extends Number>}),
 * split up the generation of weighted random walks for parallel learning. Specifically: with N threads and V vertices:
 * - First iterator generates weighted random walks starting at vertices 0 to V/N
 * - Second iterator generates weighted random walks starting at vertices V/N+1 to 2*V/N
 * - and so on
 * @param <V> Vertex type
 * @see WeightedRandomWalkIterator
 */
public class WeightedRandomWalkGraphIteratorProvider<V> implements GraphWalkIteratorProvider<V> {

    private IGraph<V, ? extends Number> graph;
    private int walkLength;
    private Random rng;
    private NoEdgeHandling mode;

    public WeightedRandomWalkGraphIteratorProvider(IGraph<V, ? extends Number> graph, int walkLength) {
        this(graph, walkLength, System.currentTimeMillis(), NoEdgeHandling.EXCEPTION_ON_DISCONNECTED);
    }

    public WeightedRandomWalkGraphIteratorProvider(IGraph<V, ? extends Number> graph, int walkLength, long seed,
                    NoEdgeHandling mode) {
        this.graph = graph;
        this.walkLength = walkLength;
        this.rng = new Random(seed);
        this.mode = mode;
    }


    @Override
    public List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators) {
        int nVertices = graph.numVertices();
        if (numIterators > nVertices)
            numIterators = nVertices;

        int verticesPerIter = nVertices / numIterators;

        List<GraphWalkIterator<V>> list = new ArrayList<>(numIterators);
        int last = 0;
        for (int i = 0; i < numIterators; i++) {
            int from = last;
            int to = Math.min(nVertices, from + verticesPerIter);
            if (i == numIterators - 1)
                to = nVertices;

            GraphWalkIterator<V> iter =
                            new WeightedRandomWalkIterator<>(graph, walkLength, rng.nextLong(), mode, from, to);
            list.add(iter);
            last = to;
        }

        return list;
    }
}
