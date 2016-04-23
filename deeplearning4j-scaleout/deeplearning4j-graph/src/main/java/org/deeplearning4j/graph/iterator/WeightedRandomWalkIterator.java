package org.deeplearning4j.graph.iterator;

import org.deeplearning4j.graph.api.*;
import org.deeplearning4j.graph.exception.NoEdgesException;
import org.deeplearning4j.graph.graph.VertexSequence;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

/**Given a graph, iterate through random walks on that graph of a specified length.
 * Unlike {@link RandomWalkIterator}, the {@code WeightedRandomWalkIterator} uses the values associated with each edge
 * to determine probabilities. Weights on each edge need not be normalized.<br>
 * Because the edge values are used to determine the probabilities of selecting an edge, the {@code WeightedRandomWalkIterator}
 * can only be used on graphs with an edge type that extends the {@link java.lang.Number} class (i.e., Integer, Double, etc)<br>
 * Random walks are generated starting at every node in the graph exactly once, though the order of the starting nodes
 * is randomized.
 * @author Alex Black
 */
public class WeightedRandomWalkIterator<V> implements GraphWalkIterator<V> {

    private final IGraph<V,? extends Number> graph;
    private final int walkLength;
    private final NoEdgeHandling mode;
    private final int firstVertex;
    private final int lastVertex;


    private int position;
    private Random rng;
    private int[] order;

    public WeightedRandomWalkIterator(IGraph<V, ? extends Number> graph, int walkLength){
        this(graph,walkLength,System.currentTimeMillis(), NoEdgeHandling.EXCEPTION_ON_DISCONNECTED);
    }

    /**Construct a RandomWalkIterator for a given graph, with a specified walk length and random number generator seed.<br>
     * Uses {@code NoEdgeHandling.EXCEPTION_ON_DISCONNECTED} - hence exception will be thrown when generating random
     * walks on graphs with vertices containing having no edges, or no outgoing edges (for directed graphs)
     * @see #WeightedRandomWalkIterator(IGraph, int, long, NoEdgeHandling)
     */
    public WeightedRandomWalkIterator(IGraph<V, ? extends Number> graph, int walkLength, long rngSeed){
        this(graph, walkLength, rngSeed, NoEdgeHandling.EXCEPTION_ON_DISCONNECTED);
    }

    /**
     * @param graph IGraph to conduct walks on
     * @param walkLength length of each walk. Walk of length 0 includes 1 vertex, walk of 1 includes 2 vertices etc
     * @param rngSeed seed for randomization
     * @param mode mode for handling random walks from vertices with either no edges, or no outgoing edges (for directed graphs)
     */
    public WeightedRandomWalkIterator(IGraph<V, ? extends Number> graph, int walkLength, long rngSeed, NoEdgeHandling mode){
        this(graph,walkLength,rngSeed,mode,0,graph.numVertices());
    }

    /**Constructor used to generate random walks starting at a subset of the vertices in the graph. Order of starting
     * vertices is randomized within this subset
     * @param graph IGraph to conduct walks on
     * @param walkLength length of each walk. Walk of length 0 includes 1 vertex, walk of 1 includes 2 vertices etc
     * @param rngSeed seed for randomization
     * @param mode mode for handling random walks from vertices with either no edges, or no outgoing edges (for directed graphs)
     * @param firstVertex first vertex index (inclusive) to start random walks from
     * @param lastVertex last vertex index (exclusive) to start random walks from
     */
    public WeightedRandomWalkIterator(IGraph<V, ? extends Number> graph, int walkLength, long rngSeed, NoEdgeHandling mode, int firstVertex,
                                      int lastVertex){
        this.graph = graph;
        this.walkLength = walkLength;
        this.rng = new Random(rngSeed);
        this.mode = mode;
        this.firstVertex = firstVertex;
        this.lastVertex = lastVertex;

        order = new int[lastVertex-firstVertex];
        for( int i=0; i<order.length; i++ ) order[i] = firstVertex+i;
        reset();
    }

    @Override
    public IVertexSequence<V> next() {
        if(!hasNext()) throw new NoSuchElementException();
        //Generate a weighted random walk starting at vertex order[current]
        int currVertexIdx = order[position++];
        int[] indices = new int[walkLength+1];
        indices[0] = currVertexIdx;
        if(walkLength == 0) return new VertexSequence<>(graph,indices);

        for( int i=1; i<=walkLength; i++ ) {
            List<? extends Edge<? extends Number>> edgeList = graph.getEdgesOut(currVertexIdx);

            //First: check if there are any outgoing edges from this vertex. If not: handle the situation
            if(edgeList == null || edgeList.size() == 0){
                switch (mode) {
                    case SELF_LOOP_ON_DISCONNECTED:
                        for (int j = i; j < walkLength; j++) indices[j] = currVertexIdx;
                        return new VertexSequence<>(graph, indices);
                    case EXCEPTION_ON_DISCONNECTED:
                        throw new NoEdgesException("Cannot conduct random walk: vertex " + currVertexIdx + " has no outgoing edges. "
                                + " Set NoEdgeHandling mode to NoEdgeHandlingMode.SELF_LOOP_ON_DISCONNECTED to self loop instead of "
                                + "throwing an exception in this situation.");
                    default:
                        throw new RuntimeException("Unknown/not implemented NoEdgeHandling mode: " + mode);
                }
            }

            //To do a weighted random walk: we need to know total weight of all outgoing edges
            double totalWeight = 0.0;
            for (Edge<? extends Number> edge : edgeList) {
                totalWeight += edge.getValue().doubleValue();
            }

            double d = rng.nextDouble();
            double threshold = d * totalWeight;
            double sumWeight = 0.0;
            for (Edge<? extends Number> edge : edgeList) {
                sumWeight += edge.getValue().doubleValue();
                if (sumWeight >= threshold) {
                    if (edge.isDirected()) {
                        currVertexIdx = edge.getTo();
                    } else {
                        if (edge.getFrom() == currVertexIdx) {
                            currVertexIdx = edge.getTo();
                        } else {
                            currVertexIdx = edge.getFrom(); //Undirected edge: might be next--currVertexIdx instead of currVertexIdx--next
                        }
                    }
                    indices[i] = currVertexIdx;
                    break;
                }
            }
        }
        return new VertexSequence<>(graph,indices);
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
