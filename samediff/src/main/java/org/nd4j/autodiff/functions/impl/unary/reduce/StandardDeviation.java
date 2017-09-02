package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class StandardDeviation  extends AbstractReduceUnaryFunction {
    protected boolean biasCorrected;

    public StandardDeviation(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public StandardDeviation(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] dimensions,boolean biasCorrected) {
        super(sameDiff, i_v, dimensions);
        this.biasCorrected = biasCorrected;
    }


    @Override
    public ArrayField doGetValue() {
        return a().std(arg().doGetValue(),
                biasCorrected ,
                dimensions);
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.StandardDeviation().name();
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        int inputs = f().getInputLength(i_v1.get(0));
        DifferentialFunction<ArrayField> g =  f().doRepeat(this,i_v1.get(0),dimensions);
        DifferentialFunction<ArrayField> ret = g.mul(arg().sub(f().mean(arg(),dimensions))).div(f()
                .one(g.getResultShape()).mul(inputs));
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }
}
