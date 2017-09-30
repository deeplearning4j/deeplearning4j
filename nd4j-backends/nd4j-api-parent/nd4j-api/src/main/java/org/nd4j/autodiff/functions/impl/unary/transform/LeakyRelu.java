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

    public LeakyRelu(SameDiff sameDiff, DifferentialFunction i_v,double cutoff) {
        super(sameDiff, i_v, new Object[]{cutoff});
        this.cutoff = cutoff;
    }

    public LeakyRelu(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, inPlace);
        this.cutoff = cutoff;
    }

    @Override
    public ArrayField doGetValue() {
        return a().leakyRelu(arg().getValue(true),cutoff);
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().leakyReluDerivative(arg(),i_v.get(0) , cutoff);

        return Collections.singletonList(ret);
    }


    @Override
    public String functionName() {
        return new LeakyReLU().name();
    }
}
