package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;


/**
 * Negative operation
 */
public class Negative extends AbstractUnaryFunction {

    public Negative(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace) {
        super(sameDiff,i_v,new Object[]{inPlace});
    }


    public Negative(SameDiff sameDiff, DifferentialFunction i_v) {
        this(sameDiff,i_v,false);
    }

    @Override
    public ArrayField doGetValue() {
        return arg().getValue(true).negate();
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        return Arrays.asList(f().neg(arg().diff(i_v).get(0)));
    }

    @Override
    public String toString() {
        return "-" + arg().toString();
    }

    @Override
    public String doGetFormula(List<Variable> variables) {
        return "-" + arg().doGetFormula(variables);
    }

    @Override
    public String functionName() {
        return new  org.nd4j.linalg.api.ops.impl.transforms.Negative().name();
    }


}
