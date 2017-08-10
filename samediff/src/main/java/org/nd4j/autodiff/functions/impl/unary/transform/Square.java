package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;

public class Square extends AbstractUnaryFunction<ArrayField> {

    public Square(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().square(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.pow(arg().getReal(), 2);
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return arg().mul(sameDiff.getFunctionFactory().val(sameDiff.getArrayFactory().one(getResultShape()).mul(2L)))
                .mul(arg().diff(i_v));
    }


    @Override
    public String functionName() {
        return new Pow().name();
    }
}
