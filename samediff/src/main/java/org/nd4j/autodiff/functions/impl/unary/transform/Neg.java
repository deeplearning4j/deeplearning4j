package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Negative;

import java.util.Collections;
import java.util.List;

public class Neg extends AbstractUnaryFunction<ArrayField> {
    public Neg(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().neg(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return -arg().getReal();
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        validateDifferentialFunctionsameDiff(arg());
        return Collections.singletonList(arg().negate());
    }


    @Override
    public String functionName() {
        return new Negative().name();
    }
}
