package org.deeplearning4j.linalg.ops.elementwise;


import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class AddOp extends BaseTwoArrayElementWiseOp {



    /**
     * The transformation for a given value
     *
     * @param value the value to applyTransformToOrigin
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray value,int i) {
        return value.addi(getFromOrigin(i));
    }


}
