package org.nd4j.autodiff.functions.impl.binary.transform.gradient;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

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
        return a().sigmoidDerivative(larg().getValue(true),rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative().name();
    }
}
