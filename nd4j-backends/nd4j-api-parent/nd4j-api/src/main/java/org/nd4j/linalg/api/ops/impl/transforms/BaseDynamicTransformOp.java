package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.List;

public abstract class BaseDynamicTransformOp extends DynamicCustomOp {

    public BaseDynamicTransformOp() {}

    public BaseDynamicTransformOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public BaseDynamicTransformOp(INDArray[] inputs, INDArray[] outputs) {
        super(null, inputs, outputs);
    }


    @Override
    public List<int[]> calculateOutputShape() {
        val args = args();
        val firstArgShape = args[0].getShape();
        val secondArgShape = args[1].getShape();
        if(args[0].getShape() == null) {
            throw new ND4JIllegalStateException("No shape found for variable " + args[0].getVarName());
        }
        if(args[1].getShape() == null) {
            throw new ND4JIllegalStateException("No shape found for variable " + args[0].getVarName());
        }


        val firstLength = ArrayUtil.prod(firstArgShape);
        val secondLength = ArrayUtil.prod(secondArgShape);
        if(firstLength > secondLength)
            return Arrays.asList(args[0].getShape());
        else
            return Arrays.asList(args[1].getShape());
    }
}
