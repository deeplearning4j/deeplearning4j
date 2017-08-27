package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;

import java.util.Collections;
import java.util.List;

public class Relu extends AbstractUnaryFunction<ArrayField> {
    public Relu(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    @Override
    public ArrayField doGetValue() {
        return a().relu(arg().getValue(true));
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(f().val(a().step(arg().getValue(true))));
    }

    @Override
    public String functionName() {
        return new RectifedLinear().name();
    }
}
