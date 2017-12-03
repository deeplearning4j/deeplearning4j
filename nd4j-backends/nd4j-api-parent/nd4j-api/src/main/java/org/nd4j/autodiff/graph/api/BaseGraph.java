package org.nd4j.autodiff.graph.api;

import org.nd4j.linalg.collection.IntArrayKeyMap;

public abstract class BaseGraph<V, E> implements IGraph<V, E> {
    //a small index and accompany reverse lookup for combinations of inputs and outputs
    //representing 1 multi input/output edge as a "node"
    protected IntArrayKeyMap<int[]> fromTo;
    protected  IntArrayKeyMap<int[]> toFrom;

    public BaseGraph() {
        fromTo = new IntArrayKeyMap<>();
        toFrom = new IntArrayKeyMap<>();
    }

    public void addEdge(int[] from, int[] to, E value, boolean directed) {
        addEdge(new Edge<>(from, to, value, directed));
        fromTo.put(from,to);
        toFrom.put(to,from);
    }


    /**
     *
     * @param output
     * @return
     */
    public int[] getFromFor(int[] output) {
        return toFrom.get(output);
    }

    /**
     *
     * @param input
     * @return
     */
    public int[] getToFor(int[] input) {
        return fromTo.get(input);
    }

    public abstract int getVertexInDegree(int[] vertex);
}
