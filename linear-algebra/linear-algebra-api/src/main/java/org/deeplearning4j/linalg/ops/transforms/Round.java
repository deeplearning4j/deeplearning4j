package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 * Round a complex or real number
 *
 * @author Adam Gibson
 */
public class Round extends BaseElementWiseOp {
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
            return NDArrays.scalar(ComplexUtil.round((org.deeplearning4j.linalg.api.complex.IComplexNumber) value.element()));
        }
        else {
            double val = (double) value.element();
            return NDArrays.scalar(Math.round(val));
        }

    }
}
