package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGradGraph;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class One<X extends Field<X>> extends Constant<X> {


    public One(TensorGradGraph graph,
               AbstractIdentityFactory<X> i_factory) {
        super(graph,i_factory.one(), i_factory);
    }




    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> dup = i_v.dup();
        if(i_v.getValue() instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v.getValue();
            addEdges(graph,dup,
                    this,
                    new MulOp().name(),
                    OpState.OpType.TRANSFORM,
                    arrayField.getInput().getShape());
        }

        return dup;
    }

    @Override
    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return mul(i_v);
    }

    @Override
    public DifferentialFunction<X> dup() {
        return new One<>(graph, getM_factory());
    }
}
