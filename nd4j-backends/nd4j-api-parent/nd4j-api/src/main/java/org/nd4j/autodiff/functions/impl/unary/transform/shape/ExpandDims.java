package org.nd4j.autodiff.functions.impl.unary.transform.shape;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class ExpandDims extends AbstractUnaryFunction {

    protected int axis;

    public ExpandDims(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs, int axis) {
        super(sameDiff, i_v, extraArgs);
        this.axis = axis;
    }

    @Override
    public ArrayField doGetValue() {
        return a().expandDims(arg().getValue(true),axis);
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        DifferentialFunction ret = f().div(arg(),f().abs(arg()));

        return Collections.singletonList(ret);
    }


    @Override
    public String functionName() {
        return "expandDims";
    }
}
