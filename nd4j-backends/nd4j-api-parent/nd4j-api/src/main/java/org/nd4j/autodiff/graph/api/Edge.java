package org.nd4j.autodiff.graph.api;

import lombok.Data;
import org.nd4j.linalg.collection.IntArrayKeyMap;

import java.util.Arrays;

/** Edge in a graph. 
 * May be a directed or undirected edge.<br>
 * Parametrized,
 * and may store a 
 * value/object associated with the edge
 */
@Data
public class Edge<T> {

    private  int[] from;
    private  int[] to;
    private  T value;
    private  boolean directed;


    public Edge(int[] from, int[] to, T value, boolean directed) {
        this.from = new IntArrayKeyMap.IntArray(from).getBackingArray();
        this.to = new IntArrayKeyMap.IntArray(to).getBackingArray();
        this.value = value;
        this.directed = directed;
    }


    @Override
    public String toString() {
        return "edge(" + (directed ? "directed" : "undirected") + "," + Arrays.toString(from) + (directed ? "->" : "--") + Arrays.toString(to) + ","
                        + (value != null ? value : "") + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        Edge<?> edge = (Edge<?>) o;

        if (directed != edge.directed) return false;
        if (!Arrays.equals(from, edge.from)) return false;
        if (!Arrays.equals(to, edge.to)) return false;
        return value != null ? value.equals(edge.value) : edge.value == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(from);
        result = 31 * result + Arrays.hashCode(to);
        result = 31 * result + (value != null ? value.hashCode() : 0);
        result = 31 * result + (directed ? 1 : 0);
        return result;
    }
}
