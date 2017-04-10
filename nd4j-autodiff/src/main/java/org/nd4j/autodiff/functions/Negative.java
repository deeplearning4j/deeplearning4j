package org.nd4j.autodiff.functions;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;


public class Negative<X extends Field<X>> extends AbstractUnaryFunction<X> {


    public Negative(Graph<NDArrayInformation,OpState> graph,DifferentialFunction<X> i_v) {
        super(graph,i_v);
    }

    @Override
    public X doGetValue() {
        return arg().getValue().negate();
    }

    @Override
    public double getReal() {
        return -arg().getReal();
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v) {
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
