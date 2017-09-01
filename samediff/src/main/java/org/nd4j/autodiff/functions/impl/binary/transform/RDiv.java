package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.AbstractBinaryFunction;
import org.nd4j.autodiff.functions.Constant;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.Variable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Not;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.RDivOp;

import java.util.Collections;
import java.util.List;

public class RDiv extends AbstractBinaryFunction<ArrayField> {
    public RDiv(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return larg().getValue(true).rdiv(rarg().getValue(true));
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        Constant<ArrayField> ym1 = f()
                .val(rarg().getValue(true).sub(a().one(getResultShape())));
        DifferentialFunction<ArrayField> ret = f().mul(rarg(),f().mul(f().pow(larg(), ym1),larg()));
        larg().setGradient(ret);
        rarg().setGradient(ret);
        return Collections.singletonList(ret);
    }



    @Override
    public String functionName() {
        return new RDivOp().name();
    }
}
