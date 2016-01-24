package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.nn.graph.ComputationGraph;

/** SubsetVertex is used to select a subset of the activations out of another GraphVertex.<br>
 * For example, a subset of the activations out of a layer.<br>
 * Note that this subset is specifying by means of an interval of the original activations.
 * For example, to get the first 10 activations of a layer (or, first 10 features out of a CNN layer) use
 * new SubsetVertex(0,9)
 * @author Alex Black
 */
@Data
public class SubsetVertex extends GraphVertex {

    private int from;
    private int to;

    /**
     * @param from The first column index, inclusive
     * @param to The last column index, inclusive
     */
    public SubsetVertex(@JsonProperty("from") int from, @JsonProperty("to") int to) {
        this.from = from;
        this.to = to;
    }

    @Override
    public SubsetVertex clone() {
        return new SubsetVertex(from,to);
    }

    @Override
    public boolean equals(Object o){
        if(!(o instanceof SubsetVertex)) return false;
        SubsetVertex s = (SubsetVertex)o;
        return s.from == from && s.to == to;
    }

    @Override
    public int hashCode(){
        return Integer.hashCode(from) ^ Integer.hashCode(to);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.SubsetVertex(graph,name,idx,from,to);
    }
}
