package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public class NormMax extends AbstractReduceUnaryFunction<ArrayField> {

    public NormMax(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().norm1(arg().doGetValue(),dimensions);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.NormMax().name();
    }



    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
        return sameDiff.getFunctionFactory().doNormGrad(this,i_v1,"normmax",dimensions);
    }
}
