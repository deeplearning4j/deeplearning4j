package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Prod extends AbstractReduceUnaryFunction<ArrayField> {
    public Prod(SameDiff sameDiff,
               DifferentialFunction<ArrayField> i_v,
               int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().prod(arg().doGetValue(),dimensions);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.Prod().name();
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        DifferentialFunction<ArrayField> ret = f().doRepeat(
                this,
                i_v1.get(0)
                ,dimensions)
                .div(f().one(getResultShape()).mul(f()
                                .getInputLength(i_v1.get(0))));
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }
}
