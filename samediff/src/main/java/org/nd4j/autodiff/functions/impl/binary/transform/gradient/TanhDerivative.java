package org.nd4j.autodiff.functions.impl.binary.transform.gradient;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class TanhDerivative extends AbstractBinaryFunction<ArrayField> {

    public TanhDerivative() {
    }

    public TanhDerivative(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2, OpState.OpType.GRADIENT);
    }

    public TanhDerivative(SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().tanhDerivative(larg().getValue(true),rarg().getValue(true));
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        return Collections.singletonList(sameDiff.getFunctionFactory().one(getResultShape()).div(sameDiff.getFunctionFactory().cosh(arg())).pow(2));
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative().name();
    }
}
