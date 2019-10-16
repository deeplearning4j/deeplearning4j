package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class CompareAndBitpack extends DynamicCustomOp {
    public CompareAndBitpack() {}

    public CompareAndBitpack(INDArray in, double threshold, INDArray out) {
        inputArguments.add(in);
        inputArguments.add(Nd4j.scalar(threshold));
        outputArguments.add(out);
    }

    public CompareAndBitpack(SameDiff sameDiff, SDVariable threshold) {
        super("", sameDiff, new SDVariable[]{threshold});
    }

    @Override
    public String opName() {
        return "compare_and_bitpack";
    }
}