package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Log extends AbstractUnaryFunction {

    public Log(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().log(arg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.log(arg().getReal());
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        validateDifferentialFunctionsameDiff(arg());
        DifferentialFunction<ArrayField> toInverse = sameDiff.setupFunction(f().div(i_v.get(0),arg()));
        arg().setGradient(toInverse);
        return Collections.singletonList(toInverse);
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Log().name();
    }
}
