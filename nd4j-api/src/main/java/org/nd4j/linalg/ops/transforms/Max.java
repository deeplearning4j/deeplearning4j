package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.BaseElementWiseOp;

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
    public Object apply(INDArray from,Object value, int i) {
        if(value instanceof IComplexNumber) {
            IComplexNumber num = (IComplexNumber) value;
            if(num.realComponent().doubleValue() > max)
                return num;
            return num.set(max,num.imaginaryComponent());
        }

        float val = (float) value;
        return (float) Math.max(max,val);
    }
}
