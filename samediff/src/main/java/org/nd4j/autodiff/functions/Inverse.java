package org.nd4j.autodiff.functions;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.samediff.SDGraph;


public class Inverse<X extends Field<X>> extends AbstractUnaryFunction<X> {

    public Inverse(SDGraph graph, DifferentialFunction<X> i_v,boolean inPlace) {
        super(graph,i_v,new Object[]{inPlace});
    }

    public Inverse(SDGraph graph, DifferentialFunction<X> i_v) {
        this(graph,i_v,false);
    }

    @Override
    public X doGetValue() {
        return arg().getValue(true).inverse();
    }

    @Override
    public double getReal() {
        return 1d / arg().getReal();
    }

    @Override
    public DifferentialFunction<X> diff(DifferentialFunction<X> i_v) {
        return new PolynomialTerm<>(graph, -1L, arg(), -2).mul(arg().diff(i_v));
    }

    @Override
    public String toString() {
        return "(" + arg().toString() + ")^(-1)";
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return "( 1d / " + arg().doGetFormula(variables) + ")";
    }

    @Override
    public String functionName() {
        return "inverse";
    }

    @Override
    public DifferentialFunction<X> inverse() {
        return arg();
    }
}
