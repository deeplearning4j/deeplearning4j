package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

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
}
