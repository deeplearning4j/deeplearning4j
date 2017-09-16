package org.nd4j.autodiff.functions;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.factory.Nd4j;


public class One extends Constant {


    public One(SameDiff sameDiff,
               int[] shape) {
        super(sameDiff,  sameDiff.a().one(shape),shape);
        this.shape = shape;
        ArrayField arrayField = m_x;
        arrayField.getInput().setScalarValue(1.0);
        this.opState = OpState.builder()
                .opName("create")
                .opNum(0)
                .result(NDArrayInformation.newInfo(Nd4j.scalar(1.0)))
                .opType(OpState.OpType.CUSTOM)
                .build();
        sameDiff.getVertexToArray().put(opState.getResult().getArrId(),Nd4j.scalar(1.0));
    }






    @Override
    public DifferentialFunction dup() {
        return new One(sameDiff, shape);
    }
}
