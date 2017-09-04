package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;

import java.util.Collections;
import java.util.List;

public class Softmax extends AbstractUnaryFunction {
    public Softmax(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return a().softmax(arg().getValue(true));
    }


    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().softmaxDerivative(arg(), i_v.get(0));
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }

    @Override
    public String functionName() {
        return new SoftMax().name();
    }
}
