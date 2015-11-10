package org.deeplearning4j.graph.graph.dl4j;

import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.api.VertexSequence;
import org.deeplearning4j.graph.exception.NoEdgesException;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;

import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Created by Alex on 9/11/2015.
 */
public class RandomWalkIterator<V> implements GraphWalkIterator<V> {

    private final SimpleGraph<V,?> graph;
    private final int walkLength;
    private final NoEdgeHandling mode;


    private int position;
    private Random rng;
    private int[] order;

    public RandomWalkIterator(SimpleGraph<V,?> graph, int walkLength ){
        this(graph,walkLength,System.currentTimeMillis(), NoEdgeHandling.EXCEPTION_ON_DISCONNECTED);
    }

    public RandomWalkIterator(SimpleGraph<V,?> graph, int walkLength, long rngSeed ){
        this(graph, walkLength, rngSeed, NoEdgeHandling.EXCEPTION_ON_DISCONNECTED);
    }

    public RandomWalkIterator(SimpleGraph<V,?> graph, int walkLength, long rngSeed, NoEdgeHandling mode ){
        this.graph = graph;
        this.walkLength = walkLength;
        this.mode = mode;

        rng = new Random(rngSeed);
        order = new int[graph.numVertices()];
        for( int i=0; i<order.length; i++ ) order[i] = i;
        reset();
    }

    @Override
    public VertexSequence<V> next() {
        if(!hasNext()) throw new NoSuchElementException();
        //Generate a random walk starting at at vertex order[current]
        int currVertexIdx = order[position++];
        int[] indices = new int[walkLength+1];
        indices[0] = currVertexIdx;
        if(walkLength == 0) return new SimpleVertexSequence<>(graph,indices);

        Vertex<V> next;
        try{
            next = graph.getRandomConnectedVertex(currVertexIdx,rng);
        }catch(NoEdgesException e){
            switch(mode){
                case SELF_LOOP_ON_DISCONNECTED:
                    for(int i=1; i<walkLength; i++) indices[i] = currVertexIdx;
                    return new SimpleVertexSequence<>(graph,indices);
                case EXCEPTION_ON_DISCONNECTED:
                    throw e;
                default:
                    throw new RuntimeException("Unknown/not implemented NoEdgeHandling mode: " + mode);
            }
        }
        indices[1] = next.vertexID();
        currVertexIdx = indices[1];

        for( int i=2; i<=walkLength; i++ ){ //<= walk length: i.e., if walk length = 2, it contains 3 vertices etc
            next = graph.getRandomConnectedVertex(currVertexIdx,rng);
            currVertexIdx = next.vertexID();
            indices[i] = currVertexIdx;
        }
        return new SimpleVertexSequence<>(graph,indices);
    }

    @Override
    public boolean hasNext() {
        return position < order.length;
    }

    @Override
    public void reset() {
        position = 0;
        //https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        for(int i=order.length-1; i>0; i-- ){
            int j = rng.nextInt(i+1);
            int temp = order[j];
            order[j] = order[i];
            order[i] = temp;
        }
    }

    @Override
    public int walkLength(){
        return walkLength;
    }
}
