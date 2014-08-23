package org.deeplearning4j.linalg.ops.transforms;


import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
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
    public INDArray apply(INDArray value, int i) {
        INDArray curr = getFromOrigin(i);
        double originValue = (double) (curr instanceof IComplexNDArray ? ((IComplexNumber) curr.element()).absoluteValue() : (double) curr.element());
        double otherValue = (double) (value instanceof IComplexNDArray ? ((IComplexNumber) value.element()).absoluteValue() : (double) value.element());
        if(originValue == otherValue) {
            if(value instanceof IComplexNDArray) {
               return NDArrays.scalar(NDArrays.createDouble(1,0));
            }
            else {
                return NDArrays.scalar(1);
            }

        }
        else {
            if(value instanceof IComplexNDArray) {
                return NDArrays.scalar(NDArrays.createDouble(0,0));
            }
            else {
                return NDArrays.scalar(0);
            }

        }
    }
}
