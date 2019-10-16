package org.nd4j.linalg.api.ops.custom;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class AdjustContrast extends BaseAdjustContrast {

    public AdjustContrast() {super();}

    public AdjustContrast(INDArray in, double factor, INDArray out) {
        super(in, factor, out);
    }

    @Override
    public String opName() {
        return "adjust_contrast";
    }
}