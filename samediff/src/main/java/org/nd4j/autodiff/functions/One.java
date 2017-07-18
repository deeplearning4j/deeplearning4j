package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiffGraph;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class One<X extends Field<X>> extends Constant<X> {


    public One(SameDiffGraph graph,
               int[] shape,
               AbstractIdentityFactory<X> i_factory) {
        super(graph,i_factory.one(shape),shape, i_factory);
        this.shape = shape;
    }




    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> dup = i_v.dup();
        if(i_v.getValue() instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v.getValue();
            addEdges(graph,
                    dup,
                    this,
                    new MulOp().name(),
                    OpState.OpType.TRANSFORM,
                    arrayField.getInput().getShape());
        }

        return dup;
    }



    @Override
    public DifferentialFunction<X> dup() {
        return new One<>(graph, shape,getM_factory());
    }
}
