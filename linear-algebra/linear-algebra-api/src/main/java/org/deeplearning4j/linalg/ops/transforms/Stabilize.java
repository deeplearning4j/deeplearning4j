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

    public Stabilize(Double k) {
        this.k = k;
    }

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
    public Object apply(INDArray from,Object value, int i) {
        double realMin =  1.1755e-38;
        double cutOff = FastMath.log(realMin);
        if(value instanceof IComplexNumber) {
            IComplexNumber c = (IComplexNumber) value;
            float curr = c.realComponent().floatValue();
            if(curr * k > -cutOff)
                return NDArrays.createDouble(-cutOff / k,c.imaginaryComponent().floatValue());
            else if(curr * k < cutOff)
                return NDArrays.createDouble(cutOff / k,c.imaginaryComponent().floatValue());


        }
        else {
            float curr = (float) value;
            if(curr * k > -cutOff)
                return -cutOff / k;
            else if(curr * k < cutOff)
                return cutOff / k;

        }



        return value;
    }
}
