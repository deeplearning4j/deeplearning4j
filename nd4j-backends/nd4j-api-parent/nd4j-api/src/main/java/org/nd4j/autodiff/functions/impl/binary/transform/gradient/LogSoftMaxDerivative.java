package org.nd4j.autodiff.functions.impl.binary.transform.gradient;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

/**
 *
 */
public class LogSoftMaxDerivative extends AbstractBinaryFunction {

    public LogSoftMaxDerivative() {
    }

    public LogSoftMaxDerivative(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2, OpState.OpType.GRADIENT);
    }

    public LogSoftMaxDerivative(SameDiff sameDiff) {
        super(sameDiff);
    }

    @Override
    public ArrayField doGetValue() {
        return a().softmaxDerivative(larg().getValue(true),rarg().getValue(true));
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Arrays.asList(f().sub(i_v.get(0),f().sum(f().exp(larg()),1)));
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.gradient.LogSoftMaxDerivative().name();
    }


}
