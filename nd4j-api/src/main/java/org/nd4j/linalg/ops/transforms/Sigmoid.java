package org.nd4j.linalg.ops.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 * Sigmoid operation
 * @author Adam Gibson
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
    public Object apply(INDArray from,Object input, int i) {
        if (input instanceof IComplexNumber) {
            IComplexNumber number = (IComplexNumber) input;
            double arg = number.complexArgument().doubleValue();
            double sigArg = 1 / 1 + (FastMath.exp(-arg)) - 1 + .5f;
            double ret =  Math.exp(sigArg);
            return Nd4j.createDouble(ret, 0);

        } else {
            double inputf = (double) input;
            double val = 1 / (1 + FastMath.exp(-inputf));
            return val;
        }
    }
}
