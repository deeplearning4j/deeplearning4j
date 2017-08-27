package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class ACosh extends AbstractUnaryFunction<ArrayField> {

    public ACosh(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().acosh(arg().getValue(true));
    }

    @Override
    public double getReal() {
        throw new IllegalStateException("");
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(sameDiff.getFunctionFactory().one(getResultShape()).div(
                sameDiff.getFunctionFactory().sqrt(arg().sub(sameDiff.getFunctionFactory().one(getResultShape())))
                        .mul(sameDiff.getFunctionFactory().sqrt(arg().add(sameDiff.getFunctionFactory().one(getResultShape()))))));
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.ACosh().name();
    }
}
