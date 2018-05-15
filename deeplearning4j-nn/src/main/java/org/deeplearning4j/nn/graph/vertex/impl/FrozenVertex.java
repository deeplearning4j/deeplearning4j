package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.graph.vertex.BaseWrapperVertex;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;

public class FrozenVertex extends BaseWrapperVertex {
    public FrozenVertex(GraphVertex underlying) {
        super(underlying);
    }
}
