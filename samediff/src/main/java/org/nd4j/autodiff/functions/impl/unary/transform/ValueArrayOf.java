package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.One;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class ValueArrayOf extends AbstractUnaryFunction {
    public ValueArrayOf(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, Object[] extraArgs) {
        super(sameDiff, i_v, shape, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return a().valueArrayOf(arg().getValue(true),shape);
    }



    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction grad = sameDiff.setupFunction(new One(sameDiff,i_v.get(0).getResultShape()));
        arg().setGradient(grad);
        return Collections.singletonList(grad);
    }


    @Override
    public String functionName() {
        return "identity";
    }
}
