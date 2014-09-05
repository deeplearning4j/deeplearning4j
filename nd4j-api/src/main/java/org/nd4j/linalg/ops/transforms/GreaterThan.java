package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 * Greater than operator
 *
 * @author Adam Gibson
 */
public class GreaterThan extends BaseElementWiseOp {
    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public Object apply(INDArray from,Object value, int i) {
        Object curr = getFromOrigin(from,i);
        float originValue = (float) (curr instanceof IComplexNDArray ? ((IComplexNumber) curr).absoluteValue() : (float) curr);
        float otherValue = (float) (value instanceof IComplexNDArray ? ((IComplexNumber) value).absoluteValue() : (float) value);
        if(originValue > otherValue) {
            if(value instanceof IComplexNumber) {
                return  NDArrays.createDouble(1, 0);
            }
            else
                return (float)  1;


        }
        else {
            if(value instanceof IComplexNumber)
                return NDArrays.createDouble(0,0);

            else
                return (float) 0;


        }
    }
}
