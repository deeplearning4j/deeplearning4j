package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class ASin extends AbstractUnaryFunction {

    public ASin(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().asin(arg().getValue(true));
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        DifferentialFunction<ArrayField> ret = f().one(getResultShape()).div(
                f().sqrt(f().one(getResultShape()).sub(arg().pow(2))));
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }



    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.ASin().name();
    }
}
