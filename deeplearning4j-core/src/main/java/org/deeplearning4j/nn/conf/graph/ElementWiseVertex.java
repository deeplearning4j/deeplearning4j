package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class ElementWiseVertex extends GraphVertex {

    public ElementWiseVertex(@JsonProperty("op") Op op) {
        this.op = op;
    }

    public enum Op {Add, Subtract, Product};

    protected Op op;

    @Override
    public ElementWiseVertex clone() {
        return new ElementWiseVertex(op);
    }
}
