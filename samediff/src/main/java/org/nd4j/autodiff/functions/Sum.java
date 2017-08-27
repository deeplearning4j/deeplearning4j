package org.nd4j.autodiff.functions;

import java.util.List;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;


public class Sum extends AbstractBinaryFunction<ArrayField> {

    public Sum(SameDiff sameDiff, DifferentialFunction<ArrayField> i_v1, DifferentialFunction<ArrayField> i_v2) {
        super(sameDiff,i_v1, i_v2);
    }

    @Override
    public ArrayField doGetValue() {
        return larg().getValue(true).add(rarg().getValue(true));
    }

    @Override
    public double getReal() {
        return larg().getReal() + rarg().getReal();
    }

    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        return (larg() == rarg()) ? larg().diff(i_v1).mul(2L) // Field is
                                                              // commutative
                                                              // with respect to
                                                              // addition.
                : larg().diff(i_v1).add(rarg().diff(i_v1));
    }

    @Override
    public String toString() {
        return "(" + larg().toString() + "+" + rarg().toString() + ")";
    }

    @Override
    public String doGetFormula(List<Variable<ArrayField>> variables) {
        return "(" + larg().doGetFormula(variables) + "+" + rarg().doGetFormula(variables) + ")";
    }

    @Override
    public String functionName() {
        return new AddOp().name();
    }
}
