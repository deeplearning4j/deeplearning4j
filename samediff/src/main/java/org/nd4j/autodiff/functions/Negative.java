package org.nd4j.autodiff.functions;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SameDiff;


/**
 * Negative operation
 * @param <X>
 */
public class Negative<X extends Field<X>> extends AbstractUnaryFunction<X> {

    public Negative(SameDiff sameDiff, DifferentialFunction<X> i_v, boolean inPlace) {
        super(sameDiff,i_v,new Object[]{inPlace});
    }


    public Negative(SameDiff sameDiff, DifferentialFunction<X> i_v) {
        this(sameDiff,i_v,false);
    }

    @Override
    public X doGetValue() {
        return arg().getValue(true).negate();
    }

    @Override
    public double getReal() {
        return -arg().getReal();
    }

    @Override
    public List<DifferentialFunction<X>> diff(List<DifferentialFunction<X>> i_v) {
        return (arg().diff(i_v)).negate();
    }

    @Override
    public String toString() {
        return "-" + arg().toString();
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return "-" + arg().doGetFormula(variables);
    }

    @Override
    public String functionName() {
        return new  org.nd4j.linalg.api.ops.impl.transforms.Negative().name();
    }

    @Override
    public DifferentialFunction<X> negate() {
        return arg();
    }

}
