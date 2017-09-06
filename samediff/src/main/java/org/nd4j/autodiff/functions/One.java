package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;


public class One extends Constant {


    public One(SameDiff sameDiff,
               int[] shape) {
        super(sameDiff,  sameDiff.getArrayFactory().one(shape),shape);
        this.shape = shape;
        ArrayField arrayField = m_x;
        arrayField.getInput().setScalarValue(1.0);
    }






    @Override
    public DifferentialFunction dup() {
        return new One(sameDiff, shape);
    }
}
