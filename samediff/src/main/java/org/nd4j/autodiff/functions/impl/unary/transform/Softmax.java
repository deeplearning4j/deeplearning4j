package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;

public class Softmax extends AbstractUnaryFunction<ArrayField> {
    public Softmax(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().softmax(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.floor(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        DifferentialFunction<ArrayField> val = sameDiff.getFunctionFactory().val(getValue(true));
        return val.mul(sameDiff.getFunctionFactory().one(getResultShape()).sub(val)).mul(arg().diff(i_v));
    }

    @Override
    public String functionName() {
        return new SoftMax().name();
    }
}
