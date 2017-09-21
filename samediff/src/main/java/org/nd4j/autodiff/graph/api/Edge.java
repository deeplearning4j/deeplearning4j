package org.nd4j.autodiff.graph.api;

import lombok.Data;

/** Edge in a graph. 
 * May be a directed or undirected edge.<br>
 * Parametrized,
 * and may store a 
 * value/object associated with the edge
 */
@Data
public class Edge<T> {

    private  int from;
    private  int[] to;
    private  T value;
    private  boolean directed;


    public Edge(int from, int[] to, T value, boolean directed) {
        this.from = from;
        this.to = to;
        this.value = value;
        this.directed = directed;
    }


    @Override
    public String toString() {
        return "edge(" + (directed ? "directed" : "undirected") + "," + from + (directed ? "->" : "--") + to + ","
                        + (value != null ? value : "") + ")";
    }

}
