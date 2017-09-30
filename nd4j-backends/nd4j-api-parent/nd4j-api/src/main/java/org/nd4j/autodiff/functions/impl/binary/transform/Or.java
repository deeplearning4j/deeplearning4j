package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class Or extends AbstractBinaryFunction {

    public Or(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2,boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace, OpState.OpType.TRANSFORM);
    }


    public Or(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }

    @Override
    public ArrayField doGetValue() {
        return a().or(larg().getValue(true), rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Or().name();
    }
}
