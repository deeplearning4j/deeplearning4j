package org.deeplearning4j.linalg.ops.transforms;


import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;

/**
 * Equal to operator
 * @author Adam Gibson
 */
public class EqualTo extends BaseElementWiseOp {
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
        float originValue = (float) (curr instanceof IComplexNumber ? ((IComplexNumber) curr).absoluteValue() : (float) curr);
        float otherValue = (float) (value instanceof IComplexNumber ? ((IComplexNumber) value).absoluteValue() : (float) value);
        if(originValue == otherValue) {
            if(value instanceof IComplexNumber) {
               return NDArrays.createDouble(1,0);
            }
            else {
                return 1;
            }

        }
        else {
            if(value instanceof IComplexNumber) {
                return  NDArrays.createDouble(0,0);
            }
            else {
                return 0;
            }

        }
    }
}
