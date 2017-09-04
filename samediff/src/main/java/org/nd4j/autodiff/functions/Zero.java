package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class Zero<X extends Field<X>> extends Constant<X> {


    public Zero(SameDiff sameDiff, int[] shape) {
        super(sameDiff, (X) sameDiff.getArrayFactory().zero(shape),shape);
        ArrayField arrayField = (ArrayField) m_x;
        arrayField.getInput().setScalarValue(0.0);
    }

    @Override
    public DifferentialFunction<X> add(DifferentialFunction<X> i_v) {
       addEdge(new AddOp().name(),i_v);
        return i_v;
    }



    @Override
    public Zero<X> mul(DifferentialFunction<X> i_v) {
        addEdge(new MulOp().name(),i_v);
        return this;
    }



    @Override
    public Constant<X> inverse() {
        // TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public Zero<X> negate() {
        addEdge(new org.nd4j.linalg.api.ops.impl.transforms.Negative().name(),this);
        return this;
    }


    private void addEdge(String opName,DifferentialFunction<X> i_v) {
        if(i_v.getValue(true) instanceof ArrayField) {
            ArrayField x = (ArrayField) i_v.getValue(true);
            addEdges(sameDiff,
                    this,
                    i_v,
                    opName,
                    OpState.OpType.TRANSFORM,
                    x.getInput().getShape(),
                    null);

        }
    }


    @Override
    public DifferentialFunction<X> dup() {
        return new Zero<>(sameDiff,shape);
    }
}
