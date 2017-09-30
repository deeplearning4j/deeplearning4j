package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class Sqrt extends AbstractUnaryFunction {

    public Sqrt(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Sqrt(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    @Override
    public ArrayField doGetValue() {
        return a().sqrt(arg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Sqrt().name();
    }
}
