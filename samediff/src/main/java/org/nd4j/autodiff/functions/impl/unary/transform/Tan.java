package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Tan  extends AbstractUnaryFunction {

    public Tan(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }


    @Override
    public ArrayField doGetValue() {
        return a().tan(arg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }



    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Tan().name();
    }
}
