package org.nd4j.autodiff.functions.impl.binary.transform.gradient;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

public class SELUDerivative extends AbstractBinaryFunction<ArrayField> {

    public SELUDerivative() {
    }

    public SELUDerivative(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2, OpState.OpType.GRADIENT);
    }

    public SELUDerivative(SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().seluDerivative(larg().getValue(true),rarg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.abs(arg().getReal());
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        DifferentialFunction<ArrayField> ret = arg().div(f().seluDerivative(arg()));
        arg().setGradient(ret);
        return Arrays.asList(ret);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.SELUDerivative().name();
    }
}
