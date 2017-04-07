package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


public class Inverse<X extends Field<X>> extends AbstractUnaryFunction<X> {


    public Inverse(Graph<NDArrayInformation,OpState> graph, DifferentialFunction<X> i_v) {
        super(graph,i_v);
    }

    @Override
    public X getValue() {
        return arg().getValue().inverse();
    }

    @Override
    public double getReal() {
        return 1d / arg().getReal();
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v) {
        return new PolynomialTerm<X>(graph,-1L, arg(), -2).mul(arg().diff(i_v));
    }

    @Override
    public String toString() {
        return "(" + arg().toString() + ")^(-1)";
    }

    @Override
    public String getFormula(List<Variable<X>> variables) {
        return "( 1d / " + arg().getFormula(variables) + ")";
    }

    @Override
    public DifferentialFunction<X> inverse() {
        return arg();
    }
}
