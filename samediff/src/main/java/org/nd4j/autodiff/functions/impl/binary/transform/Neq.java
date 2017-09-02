package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.*;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Neq extends AbstractBinaryFunction {
    public Neq(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return a().neq(larg().getValue(true), rarg().getValue(true));
    }


    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        return Collections.singletonList(f().neg(i_v.get(0)));
    }



    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.NotEqualTo().name();
    }
}
