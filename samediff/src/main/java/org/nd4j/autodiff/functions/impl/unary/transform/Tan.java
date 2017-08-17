package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.PolynomialTerm;
import org.nd4j.autodiff.samediff.SameDiff;

public class Tan  extends AbstractUnaryFunction<ArrayField> {

    public Tan(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }


    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().tan(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.tan(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return (new PolynomialTerm<>(sameDiff,1, sameDiff.getFunctionFactory().cos(arg()), -2)).mul(arg().diff(i_v));
    }



    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Tan().name();
    }
}
