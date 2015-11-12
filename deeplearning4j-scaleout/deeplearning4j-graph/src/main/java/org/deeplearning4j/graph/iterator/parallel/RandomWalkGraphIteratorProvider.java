package org.deeplearning4j.graph.iterator.parallel;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.RandomWalkIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RandomWalkGraphIteratorProvider<V> implements GraphWalkIteratorProvider<V> {

    private IGraph<V,?> graph;
    private int walkLength;
    private Random rng;
    private NoEdgeHandling mode;

    public RandomWalkGraphIteratorProvider( IGraph<V,?> graph, int walkLength ){
        this(graph, walkLength, System.currentTimeMillis(), NoEdgeHandling.EXCEPTION_ON_DISCONNECTED);
    }

    public RandomWalkGraphIteratorProvider( IGraph<V,?> graph, int walkLength, long seed, NoEdgeHandling mode ){
        this.graph = graph;
        this.walkLength = walkLength;
//        this.seed = seed;
        this.rng = new Random(seed);
        this.mode = mode;
    }


    @Override
    public List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators) {

        int nVertices = graph.numVertices();
        if(numIterators > nVertices) numIterators = nVertices;

        int verticesPerIter = nVertices / numIterators;

        List<GraphWalkIterator<V>> list = new ArrayList<>(numIterators);
        int last = 0;
        for( int i=0; i<numIterators; i++ ){
            int from = last;
            int to = Math.min(nVertices,from+verticesPerIter);
            if(i == numIterators - 1) to = nVertices;

            GraphWalkIterator<V> iter = new RandomWalkIterator<V>(graph,walkLength,rng.nextLong(),mode,from,to);
            list.add(iter);
            last = to;
        }

        return list;
    }
}
