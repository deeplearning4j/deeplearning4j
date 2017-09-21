package org.nd4j.autodiff.opstate;

import lombok.Getter;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.samediff.SameDiff;

public class OpStateEdge extends Edge<OpState> {
    @Getter
    private SameDiff sameDiff;

    public OpStateEdge(int from, int[] to, OpState value, boolean directed) {
        super(from, to, value, directed);
        this.sameDiff = value.getDifferentialFunction().getSameDiff();

    }
}
