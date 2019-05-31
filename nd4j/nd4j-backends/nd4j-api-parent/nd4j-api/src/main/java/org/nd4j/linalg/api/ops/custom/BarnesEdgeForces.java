package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class BarnesEdgeForces extends DynamicCustomOp {

    public BarnesEdgeForces(){ }

    public BarnesEdgeForces(INDArray rowP, INDArray colP, INDArray valP, INDArray dataP, long N,
                            INDArray output) {

        inputArguments.add(rowP);
        inputArguments.add(colP);
        inputArguments.add(valP);
        inputArguments.add(dataP);

        iArguments.add(N);

        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "barnes_edge_forces";
    }
}
