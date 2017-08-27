package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class ExpandDims extends AbstractUnaryFunction<ArrayField> {

    protected int axis;

    public ExpandDims(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs, int axis) {
        super(sameDiff, i_v, extraArgs);
        this.axis = axis;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().expandDims(arg().getValue(true),axis);
    }

    @Override
    public double getReal() {
        return Math.abs(arg().getReal());
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        return Collections.singletonList(arg().div(sameDiff.getFunctionFactory().abs(arg())));
    }


    @Override
    public String functionName() {
        return "expandDims";
    }
}
