package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class ATan extends AbstractUnaryFunction {

    public ATan(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public ATan(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    @Override
    public ArrayField doGetValue() {
        return a().atan(arg().getValue(true));
    }

    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().div(f().one(getResultShape()),
                f().sqrt(f().sub(f().one(getResultShape()),f().pow(arg(),2))));
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.ATan().name();
    }
}
