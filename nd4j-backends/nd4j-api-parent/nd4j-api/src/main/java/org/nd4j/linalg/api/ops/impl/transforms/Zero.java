package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;


public class Zero extends Constant {


    public Zero(SameDiff sameDiff, int[] shape) {
        super(sameDiff, sameDiff.getArrayFactory().zero(shape),shape);
        ArrayField arrayField = m_x;
        arrayField.getInput().setScalarValue(0.0);
    }





    @Override
    public DifferentialFunction dup() {
        return new Zero(sameDiff,shape);
    }
}
