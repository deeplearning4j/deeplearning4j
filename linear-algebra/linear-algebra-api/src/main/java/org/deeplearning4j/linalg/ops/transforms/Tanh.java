package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 * Tanh transform
 * @author Adam Gibson
 */
public class Tanh extends BaseElementWiseOp {


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
            IComplexNumber element = (IComplexNumber) value.element();
            return NDArrays.scalar(ComplexUtil.tanh(element));
        }
        else  {
            double d = (double) value.element();
            return NDArrays.scalar(Math.tanh(d));
        }
    }
}
