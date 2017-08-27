package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.PolynomialTerm;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Tan  extends AbstractUnaryFunction<ArrayField> {

    public Tan(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }


    @Override
    public ArrayField doGetValue() {
        return a().tan(arg().getValue(true));
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(new PolynomialTerm<>(sameDiff,1, f().cos(arg()), -2));
    }



    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Tan().name();
    }
}
