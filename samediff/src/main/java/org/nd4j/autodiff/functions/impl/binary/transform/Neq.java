package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.Not;

import java.util.List;

public class Neq extends AbstractBinaryFunction<ArrayField> {
    public Neq(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().neq(larg().getValue(true), rarg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.pow(larg().getReal(), rarg().getReal());
    }

    @Override
    public DifferentialFunction<ArrayField> diff(DifferentialFunction<ArrayField> i_v) {
        Constant<ArrayField> ym1 = sameDiff.getFunctionFactory()
                .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
        return rarg().mul(sameDiff.getFunctionFactory().pow(larg(), ym1))
                .mul(larg().diff(i_v));
    }

    @Override
    public String toString() {
        return "neq(" + larg().toString() + ", " + rarg().toString() + ")";
    }

    @Override
    public String doGetFormula(List<Variable<ArrayField> > variables) {
        return "neq(" + larg().doGetFormula(variables) + ","
                + rarg().doGetFormula(variables) + ")";
    }

    @Override
    public String functionName() {
        return new Not().name();
    }
}
