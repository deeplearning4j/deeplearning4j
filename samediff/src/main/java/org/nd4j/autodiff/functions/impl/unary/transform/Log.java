package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Inverse;
import org.nd4j.autodiff.samediff.SameDiff;

public class Log extends AbstractUnaryFunction<ArrayField> {

    public Log(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().log(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.log(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        validateDifferentialFunctionsameDiff(arg());
        DifferentialFunction<ArrayField> toInverse = arg().div(i_v);
        return toInverse;
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Log().name();
    }
}
