package org.nd4j.autodiff.functions.impl.binary.transform.gradient;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.AbstractUnaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

public class SigmoidDerivative extends AbstractBinaryFunction<ArrayField> {

    public SigmoidDerivative() {
    }

    public SigmoidDerivative(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2, OpState.OpType.GRADIENT);
    }

    public SigmoidDerivative(SameDiff sameDiff) {
        super(sameDiff);
    }


    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().sigmoidDerivative(larg().getValue(true),rarg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.floor(arg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative().name();
    }
}
