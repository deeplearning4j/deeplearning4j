package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Prod extends AbstractReduceUnaryFunction {
    public Prod(SameDiff sameDiff,
               DifferentialFunction i_v,
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
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        DifferentialFunction ret = f().div(f().doRepeat(
                this,
                i_v1.get(0)
                ,dimensions),f().mul(f().one(getResultShape()),f()
                                .getInputLength(i_v1.get(0))));

        return Collections.singletonList(ret);
    }
}
