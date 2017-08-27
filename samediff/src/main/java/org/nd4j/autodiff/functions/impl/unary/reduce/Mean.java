package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Mean extends AbstractReduceUnaryFunction<ArrayField> {
    public Mean(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().mean(arg().doGetValue(),dimensions);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.Mean().name();
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        return Collections.singletonList(sameDiff.getFunctionFactory().doRepeat(this,i_v1.get(0),dimensions)
                .div(sameDiff.getFunctionFactory().one(i_v1.get(0).getResultShape()).mul(
                        sameDiff.getFunctionFactory().getInputLength(i_v1.get(0)))));
    }
}
