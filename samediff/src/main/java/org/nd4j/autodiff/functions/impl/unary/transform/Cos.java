package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public class Cos extends AbstractUnaryFunction<ArrayField> {
    public Cos(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().cos(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.cos(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return (sameDiff.getFunctionFactory().sin(arg()).mul(arg().diff(i_v))).negate();
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Cos().name();
    }

}
