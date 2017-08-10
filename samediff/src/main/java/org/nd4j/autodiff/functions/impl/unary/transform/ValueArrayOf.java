package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

public class ValueArrayOf extends AbstractUnaryFunction<ArrayField> {
    public ValueArrayOf(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, int[] shape, Object[] extraArgs) {
        super(sameDiff, i_v, shape, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().valueArrayOf(arg().getValue(true),shape);
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return "valueArray";
    }
}
