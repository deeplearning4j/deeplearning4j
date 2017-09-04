package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.*;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Eq extends AbstractBinaryFunction {
    public Eq(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return a().eq(larg().getValue(true), rarg().getValue(true));
    }


    @Override
    public List<DifferentialFunction> diff(List<DifferentialFunction> i_v) {
        Constant ym1 = f()
                .val(rarg().getValue(true).sub(a().one(getResultShape())));
        DifferentialFunction ret = f().mul(f().mul(rarg(),f().pow(larg(), 2.0)),larg());
        larg().setGradient(ret);
        rarg().setGradient(ret);
        return Collections.singletonList(ret);
    }


    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.comparison.EqualTo().name();
    }
}
