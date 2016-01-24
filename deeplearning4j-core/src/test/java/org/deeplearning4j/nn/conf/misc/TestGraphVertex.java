package org.deeplearning4j.nn.conf.misc;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.graph.GraphVertex;

@AllArgsConstructor @Data @NoArgsConstructor
public class TestGraphVertex extends GraphVertex {

    private int firstVal;
    private int secondVal;

    @Override
    public GraphVertex clone() {
        return new TestGraphVertex(firstVal,secondVal);
    }
}
