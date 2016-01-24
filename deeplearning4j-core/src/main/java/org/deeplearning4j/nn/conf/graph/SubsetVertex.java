package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.nn.graph.ComputationGraph;

@Data
public class SubsetVertex extends GraphVertex {

    private int from;
    private int to;


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
