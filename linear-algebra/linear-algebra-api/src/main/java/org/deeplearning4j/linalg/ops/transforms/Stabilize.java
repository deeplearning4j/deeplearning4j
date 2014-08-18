package org.deeplearning4j.linalg.ops.transforms;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;

/**
 * Ensures numerical stability.
 * Clips values of input such that
 * exp(k * in) is within single numerical precision
 *
 * @author Adam Gibson
 */
public class Stabilize extends BaseElementWiseOp {
    private double k = 1;

    public Stabilize(double k) {
        this.k = k;
    }
    public Stabilize() {
    }

    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray value, int i) {
        double realMin =  1.1755e-38;
        double cutOff = FastMath.log(realMin);
        if(value instanceof IComplexNDArray) {
            IComplexNumber c = (IComplexNumber) value.element();
            double curr = c.realComponent().doubleValue();
            if(curr * k > -cutOff)
                return NDArrays.scalar(NDArrays.createDouble(-cutOff / k,c.imaginaryComponent().doubleValue()));
            else if(curr * k < cutOff)
                return NDArrays.scalar(NDArrays.createDouble(cutOff / k,c.imaginaryComponent().doubleValue()));


        }
        else {
            double curr = (double) value.element();
            if(curr * k > -cutOff)
                return NDArrays.scalar(-cutOff / k);
            else if(curr * k < cutOff)
                return NDArrays.scalar(cutOff / k);

        }



        return null;
    }
}
