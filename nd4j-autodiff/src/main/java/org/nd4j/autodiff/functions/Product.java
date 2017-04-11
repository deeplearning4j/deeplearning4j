package org.nd4j.autodiff.functions;

import java.util.List;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class Product<X extends Field<X>> extends AbstractBinaryFunction<X> {

    public Product(Graph<NDArrayInformation,OpState> graph, DifferentialFunction<X> i_v1, DifferentialFunction<X> i_v2) {
        super(graph,i_v1, i_v2);
    }




    @Override
    public X doGetValue() {
        return larg().getValue().mul(rarg().getValue());
    }

    @Override
    public double getReal() {
        return larg().getReal() * rarg().getReal();
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v1) {
        return (larg().equals(rarg())) ? larg().diff(i_v1).mul(rarg()).mul(2L) // Field
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
    public String doGetFormula(List<Variable<X>> variables) {
        return "(" + larg().doGetFormula(variables) + "*" + rarg().doGetFormula(variables) + ")";
    }

    @Override
    public String functionName() {
        return new MulOp().name();
    }
}
