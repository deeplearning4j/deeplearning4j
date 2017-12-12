package org.nd4j.autodiff.opstate;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.graph.api.Edge;

public class EdgeId extends Edge<DifferentialFunction> {
    public EdgeId(int[] from, int[] to, DifferentialFunction value, boolean directed) {
        super(from, to, value, directed);

    }
}
