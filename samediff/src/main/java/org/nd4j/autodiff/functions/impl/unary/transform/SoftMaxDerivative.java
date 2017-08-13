package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;

public class SoftMaxDerivative extends AbstractUnaryFunction<ArrayField> {

    public SoftMaxDerivative(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().softmaxDerivative(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.floor(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return sameDiff.getFunctionFactory().softmaxDerivative(i_v).mul(arg().diff(i_v));
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative().name();
    }
}
