package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 *  Absolute value
 */
public class Abs extends BaseElementWiseOp {
    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray value, int i) {
       if(value instanceof IComplexNDArray) {
           return NDArrays.scalar(ComplexUtil.abs((org.deeplearning4j.linalg.api.complex.IComplexNumber) value.element()));
       }
       return NDArrays.scalar(Math.abs((float) value.element()));
    }
}
