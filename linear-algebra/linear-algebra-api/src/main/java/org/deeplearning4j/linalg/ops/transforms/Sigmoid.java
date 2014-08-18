package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;

/**
 * Sigmoid operation
 */
public class Sigmoid extends BaseElementWiseOp {
    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param input the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray input, int i) {
        if(input instanceof IComplexNDArray) {
            IComplexNumber number = (IComplexNumber) input.element();
            double arg = number.complexArgument().doubleValue();
            double sigArg = 1  / 1 + (Math.exp(-arg)) - 1 + .5;
            double ret = Math.exp(sigArg);
            return NDArrays.scalar(NDArrays.createDouble(ret, 0));

        }
        else
            return  NDArrays.scalar(1 / 1 + Math.exp(-(double) input.element()));
    }
}
