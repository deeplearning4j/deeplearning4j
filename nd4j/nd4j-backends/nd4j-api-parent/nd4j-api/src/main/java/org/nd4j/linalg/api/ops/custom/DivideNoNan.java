package org.nd4j.linalg.api.ops.custom;

import org.apache.commons.math3.analysis.function.Divide;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class DivideNoNan extends DynamicCustomOp {
    public DivideNoNan() {
    }

    public DivideNoNan(INDArray in1, INDArray in2, INDArray out) {
        inputArguments.add(in1);
        inputArguments.add(in2);
        outputArguments.add(out);
    }

    @Override
    public String opName() {
        return "divide_no_nan";
    }
}