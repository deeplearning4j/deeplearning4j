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
}
