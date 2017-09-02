package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;

import java.util.Collections;
import java.util.List;

public class LeakyRelu  extends AbstractUnaryFunction {
    private double cutoff;

    public LeakyRelu(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v,double cutoff) {
        super(sameDiff, i_v, new Object[]{cutoff});
        this.cutoff = cutoff;
    }

    @Override
    public ArrayField doGetValue() {
        return a().leakyRelu(arg().getValue(true),cutoff);
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        DifferentialFunction<ArrayField> ret = f().leakyReluDerivative(arg(),i_v.get(0) , cutoff);
        arg().setGradient(ret);
        return Collections.singletonList(ret);
    }


    @Override
    public String functionName() {
        return new LeakyReLU().name();
    }
}
