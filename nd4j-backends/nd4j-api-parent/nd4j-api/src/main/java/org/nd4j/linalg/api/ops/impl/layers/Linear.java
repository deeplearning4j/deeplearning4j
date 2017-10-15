package org.nd4j.linalg.api.ops.impl.layers;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

public class Linear extends DynamicCustomOp {

    public Linear(SameDiff sameDiff, DifferentialFunction[] args) {
        super(null, sameDiff, args);
    }

    public Linear(INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        super(null, inputs, outputs, tArguments, iArguments);
    }

    public Linear(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }

    public Linear(SameDiff sameDiff, DifferentialFunction[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    @Override
    public String opName() {
        return "linear";
    }
}
