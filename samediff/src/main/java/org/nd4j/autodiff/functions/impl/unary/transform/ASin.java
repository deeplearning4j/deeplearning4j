package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public class ASin extends AbstractUnaryFunction<ArrayField> {

    public ASin(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().asin(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.asin(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return sameDiff.getFunctionFactory().one(getResultShape()).div(sameDiff.getFunctionFactory()
                .sqrt(sameDiff.getFunctionFactory()
                .one(getResultShape()).sub(arg().pow(2))));
    }



    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.ASin().name();
    }
}
