package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;

/**
 *
 * Max function
 * @author Adam Gibson
 */
public class Max extends BaseElementWiseOp {
    private double max = 0;

    public Max(double max) {
        this.max = max;
    }

    public Max() {}

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
            IComplexNumber num = (IComplexNumber) value.element();
            if(num.realComponent().doubleValue() > max)
                return NDArrays.scalar(num);
            return NDArrays.scalar(num.set(max,num.imaginaryComponent()));
        }

        double val = (double) value.element();
        return NDArrays.scalar(Math.max(max,val));
    }
}
