package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Exp extends AbstractUnaryFunction {

    public Exp(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Exp(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().exp(arg().getValue(true));
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().mul(f().exp(arg()),i_v.get(0));

        return Collections.singletonList(ret);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Exp().name();
    }

}
