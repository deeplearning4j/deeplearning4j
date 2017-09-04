package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Negative;

import java.util.Collections;
import java.util.List;

public class Neg extends AbstractUnaryFunction {
    public Neg(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return a().neg(arg().getValue(true));
    }


    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        validateDifferentialFunctionsameDiff(arg());
        DifferentialFunction ret = f().neg(arg());
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }


    @Override
    public String functionName() {
        return new Negative().name();
    }
}
