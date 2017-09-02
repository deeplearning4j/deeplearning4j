package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class One<X extends Field<X>> extends Constant {


    public One(SameDiff sameDiff,
               int[] shape) {
        super(sameDiff,  sameDiff.getArrayFactory().one(shape),shape);
        this.shape = shape;
        ArrayField arrayField = (ArrayField) m_x;
        arrayField.getInput().setScalarValue(1.0);
    }




    @Override
    public DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> i_v) {
        DifferentialFunction<ArrayField> dup = i_v.dup();
        ArrayField arrayField = i_v.getValue(true);
        addEdges(sameDiff,
                dup,
                this,
                new MulOp().name(),
                OpState.OpType.TRANSFORM,
                arrayField.getInput().getShape());


        return dup;
    }



    @Override
    public DifferentialFunction<ArrayField> dup() {
        return new One<>(sameDiff, shape);
    }
}
