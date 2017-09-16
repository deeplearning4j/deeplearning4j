package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Sum extends AbstractReduceUnaryFunction {
    public Sum(SameDiff sameDiff,
               DifferentialFunction i_v,
               int[] dimensions) {
        super(sameDiff, i_v, dimensions);
        validateDifferentialFunctionsameDiff(i_v);
    }

    @Override
    public ArrayField doGetValue() {
        return a().sum(arg().doGetValue(),dimensions);
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.Sum().name();
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        validateFunctionReference(i_v1);
        DifferentialFunction repeat =  f().doRepeat(
                i_v1.get(0),
                arg(),dimensions);
        arg().setGradient(repeat);
        return Collections.singletonList(repeat);
    }
}
