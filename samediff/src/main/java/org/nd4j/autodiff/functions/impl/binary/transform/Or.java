package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.*;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

public class Or extends AbstractBinaryFunction {
    public Or(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return a().or(larg().getValue(true), rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        throw new UnsupportedOperationException();
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Or().name();
    }
}
