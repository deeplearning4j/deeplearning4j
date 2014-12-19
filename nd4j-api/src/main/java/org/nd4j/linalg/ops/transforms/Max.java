package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 *
 * Max function. (The max of zero and a number)
 * @author Adam Gibson
 */
public class Max extends BaseElementWiseOp {
    private Number max = 0;
    public Max(Double max) {
        this.max = max;
    }

    public Max(Float max) {
        this.max = max;
    }

    public Max(Number max) {
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
            if(num.realComponent().doubleValue() > max.doubleValue())
                return num;
            return num.set(max,num.imaginaryComponent());
        }

        double val = (double) value;
        return  Math.max(max.doubleValue(),val);
    }
}
