package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class One<X extends Field<X>> extends Constant<X> {


    public One(Graph<NDArrayInformation,OpState> graph,AbstractIdentityFactory<X> i_factory) {
        super(graph,i_factory.one(), i_factory);
    }




    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
       addEdge(new MulOp().name());
        return i_v;
    }

    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        addEdge(new MulOp().name());
        return i_v;
    }

}
