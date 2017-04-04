package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.Field;


public class Product<X extends Field<X>> extends AbstractBinaryFunction<X> {

    public Product(DifferentialFunction<X> i_v1, DifferentialFunction<X> i_v2) {
        super(i_v1, i_v2);
    }

    @Override
    public X getValue() {
        return larg().getValue().mul(rarg().getValue());
    }

    @Override
    public double getReal() {
        return larg().getReal() * rarg().getReal();
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v1) {
        return (larg() == rarg()) ? larg().diff(i_v1).mul(rarg()).mul(2L) // Field
                                                                          // is
                                                                          // commutative
                                                                          // with
                                                                          // respect
                                                                          // to
                                                                          // multiplication.
                : (larg().diff(i_v1).mul(rarg())).plus(larg().mul(rarg().diff(i_v1)));
    }

    @Override
    public String toString() {
        return "(" + larg().toString() + "*" + rarg().toString() + ")";
    }

    @Override
    public String getFormula(List<Variable<X>> variables) {
        return "(" + larg().getFormula(variables) + "*" + rarg().getFormula(variables) + ")";
    }
}
