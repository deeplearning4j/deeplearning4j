package org.nd4j.autodiff.opstate;

import org.nd4j.autodiff.graph.api.Edge;

public class OpStateEdge extends Edge<OpState> {
    public OpStateEdge(int[] from, int[] to, OpState value, boolean directed) {
        super(from, to, value, directed);

    }
}
