package org.nd4j.autodiff.opstate;

import org.nd4j.autodiff.graph.api.Edge;

public class EdgeId extends Edge<String> {
    public EdgeId(int[] from, int[] to, String value, boolean directed) {
        super(from, to, value, directed);

    }
}
