package org.nd4j.autodiff.functions.impl.binary.transform.gradient;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

public class HardTanhDerivative extends AbstractBinaryFunction {
    public HardTanhDerivative() {
    }

    public HardTanhDerivative(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2, OpState.OpType.GRADIENT);
    }

    public HardTanhDerivative(SameDiff sameDiff) {
        super(sameDiff);
    }
    @Override
    public ArrayField doGetValue() {
        return a().hardTanhDerivative(larg().getValue(true),rarg().getValue(true));
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        DifferentialFunction ret = f().one(getResultShape());
        arg().setGradient(ret);
        return Arrays.asList(ret);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.HardTanhDerivative().name();
    }


}
