package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class Zero extends Constant {


    public Zero(SameDiff sameDiff, int[] shape) {
        super(sameDiff, sameDiff.getArrayFactory().zero(shape),shape);
        ArrayField arrayField = m_x;
        arrayField.getInput().setScalarValue(0.0);
    }

    @Override
    public DifferentialFunction<ArrayField> add(DifferentialFunction<ArrayField> i_v) {
        addEdge(new AddOp().name(),i_v);
        return i_v;
    }



    @Override
    public Zero mul(DifferentialFunction<ArrayField> i_v) {
        addEdge(new MulOp().name(),i_v);
        return this;
    }



    @Override
    public Constant inverse() {
        // TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public Zero negate() {
        addEdge(new org.nd4j.linalg.api.ops.impl.transforms.Negative().name(),this);
        return this;
    }


    private void addEdge(String opName,DifferentialFunction<ArrayField> i_v) {
        ArrayField x = i_v.getValue(true);
        addEdges(sameDiff,
                this,
                i_v,
                opName,
                OpState.OpType.TRANSFORM,
                x.getInput().getShape(),
                null);


    }


    @Override
    public DifferentialFunction<ArrayField> dup() {
        return new Zero(sameDiff,shape);
    }
}
