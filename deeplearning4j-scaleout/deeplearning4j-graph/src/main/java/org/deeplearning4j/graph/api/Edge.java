package org.deeplearning4j.graph.api;

import lombok.Data;

@Data
public class Edge<T> {

    private final int from;
    private final int to;
    private final T value;
    private final boolean directed;

    public Edge(int from, int to, T value, boolean directed ){
        this.from = from;
        this.to = to;
        this.value = value;
        this.directed = directed;
    }

    @Override
    public String toString() {
        return "edge(" + (directed ? "directod" : "undirected") + "," + from+  (directed ? "->" : "--") + to + ","
                + (value!=null ? value : "") + ")";
    }
}
