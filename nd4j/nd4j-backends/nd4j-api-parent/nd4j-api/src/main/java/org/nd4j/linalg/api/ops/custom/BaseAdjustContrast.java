package org.nd4j.linalg.api.ops.custom;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public abstract class BaseAdjustContrast extends DynamicCustomOp {
    public BaseAdjustContrast() {
    }

    public BaseAdjustContrast(INDArray in, double factor, INDArray out) {
        Preconditions.checkArgument(in.rank() >= 3,
                String.format("AdjustContrast: op expects rank of input array to be >= 3, but got %d instead", in.rank()));
        inputArguments.add(in);
        outputArguments.add(out);

        addTArgument(factor);
    }
}