package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.One;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

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
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(new One<>(sameDiff,i_v.get(0).getResultShape()));
    }


    @Override
    public String functionName() {
        return "identity";
    }
}
