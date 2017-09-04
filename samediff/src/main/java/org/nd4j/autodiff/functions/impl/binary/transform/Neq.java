package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Not;

import java.util.Collections;
import java.util.List;

public class Neq extends AbstractBinaryFunction<ArrayField> {
    public Neq(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return a().neq(larg().getValue(true), rarg().getValue(true));
    }


    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        Constant<ArrayField> ym1 = sameDiff.getFunctionFactory()
                .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
        DifferentialFunction<ArrayField> ret = rarg().mul(sameDiff.getFunctionFactory().pow(larg(), ym1))
                .mul(larg());
        larg().setGradient(ret);
        rarg().setGradient(ret);
        return Collections.singletonList(ret);
    }



    @Override
    public String functionName() {
        return new Not().name();
    }
}
