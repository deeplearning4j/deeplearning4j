package org.nd4j.autodiff.functions.impl.binary.transform;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.*;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Collections;
import java.util.List;

public class Pow extends AbstractBinaryFunction<ArrayField> {
    public Pow(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().pow(larg().getValue(true), rarg().getValue(true));
    }

    @Override
    public double getReal() {
        return Math.pow(larg().getReal(), rarg().getReal());
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v) {
        Constant<ArrayField> ym1 = sameDiff.getFunctionFactory()
                .val(rarg().getValue(true).sub(sameDiff.getArrayFactory().one(getResultShape())));
        return Collections.singletonList(rarg().mul(sameDiff.getFunctionFactory().pow(larg(), ym1))
                .mul(larg()));
    }

    @Override
    public String toString() {
        return "pow(" + larg().toString() + ", " + rarg().toString() + ")";
    }

    @Override
    public String doGetFormula(List<Variable<ArrayField> > variables) {
        return "pow(" + larg().doGetFormula(variables) + ","
                + rarg().doGetFormula(variables) + ")";
    }

    @Override
    public String functionName() {
        return new org.nd4j.linalg.api.ops.impl.transforms.Pow().name();
    }
}
