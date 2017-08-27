package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;

import java.util.Collections;
import java.util.List;

public class Softmax extends AbstractUnaryFunction<ArrayField> {
    public Softmax(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return a().softmax(arg().getValue(true));
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(f().softmaxDerivative(arg(), i_v.get(0)));
    }

    @Override
    public String functionName() {
        return new SoftMax().name();
    }
}
