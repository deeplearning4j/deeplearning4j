package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class BarnesHutSymmetrize extends DynamicCustomOp {

    public BarnesHutSymmetrize(INDArray rowP, INDArray colP, INDArray valP, long N,
                               INDArray output) {
        inputArguments.add(rowP);
        inputArguments.add(colP);
        inputArguments.add(valP);

        iArguments.add(N);

        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "barnes_symmetrized";
    }
}
