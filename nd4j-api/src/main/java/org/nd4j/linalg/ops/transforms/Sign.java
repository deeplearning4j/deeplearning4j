package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 * Signum function
 * @author Adam Gibson
 */
public class Sign extends BaseElementWiseOp {
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
            IComplexNumber n = (IComplexNumber) value;
            if(n.realComponent().doubleValue() > 0)
                return Nd4j.createDouble(1,0);
            else if(n.realComponent().doubleValue() < 0)
                return Nd4j.createDouble(-1,0);
            else {
                float val = (float) apply(from,n.imaginaryComponent().doubleValue(),i);
                return Nd4j.createDouble(val,0);
            }
        }
        else {
            float n = (float) value;
            if(n < 0)
                return (float) -1;
            else if(n > 0)
                return (float) 1;
            return (float) 0;
        }

    }
}
