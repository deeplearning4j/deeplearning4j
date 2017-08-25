package org.nd4j.autodiff.functions.impl.unary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;

public class LeakyRelu  extends AbstractUnaryFunction<ArrayField> {
    private double cutoff;

    public LeakyRelu(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v,double cutoff) {
        super(sameDiff, i_v, new Object[]{cutoff});
        this.cutoff = cutoff;
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().leakyRelu(arg().getValue(true),cutoff);
    }


    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        return sameDiff.getFunctionFactory().leakyReluDerivative(arg(),i_v , cutoff);
    }


    @Override
    public String functionName() {
        return new LeakyReLU().name();
    }
}
