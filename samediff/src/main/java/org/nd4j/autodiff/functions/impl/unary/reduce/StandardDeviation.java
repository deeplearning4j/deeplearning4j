package org.nd4j.autodiff.functions.impl.unary.reduce;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractReduceUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public class StandardDeviation  extends AbstractReduceUnaryFunction<ArrayField> {
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
        return sameDiff.getArrayFactory().std(arg().doGetValue(),
                biasCorrected ,
                dimensions);
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.accum.StandardDeviation().name();
    }



    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v1) {
        validateDifferentialFunctionsameDiff(i_v1);
        int inputs = sameDiff.getFunctionFactory().getInputLength(i_v1);
        DifferentialFunction<ArrayField> g =  sameDiff.getFunctionFactory().doRepeat(this,i_v1,dimensions);
        return g.mul(arg().sub(sameDiff.getFunctionFactory().mean(arg(),dimensions))).div(sameDiff.getFunctionFactory().one(g.getResultShape()).mul(inputs));
    }
}
