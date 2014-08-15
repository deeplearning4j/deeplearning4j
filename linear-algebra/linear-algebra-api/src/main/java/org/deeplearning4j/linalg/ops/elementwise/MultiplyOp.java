package org.deeplearning4j.linalg.ops.elementwise;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Multiply a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class MultiplyOp extends BaseTwoArrayElementWiseOp {

    /**
     * The transformation for a given value
     *
     * @param value the value to applyTransformToOrigin
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray value,int i) {
        if(!value.isScalar())
            throw new IllegalArgumentException("Unable to access individual element with a scalar");
        INDArray origin = getFromOrigin(i);
        if(value instanceof IComplexNDArray) {
            IComplexNDArray complexValue = (IComplexNDArray) value;
            IComplexNumber otherValue = (IComplexNumber) complexValue.element();
            //complex + complex
            if(origin instanceof IComplexNDArray) {
                IComplexNDArray originComplex = (IComplexNDArray) origin;
                IComplexNumber originValue = (IComplexNumber) originComplex.element();
                return NDArrays.scalar(originValue.mul(otherValue));
            }

            //real + complex
            else {
                double element = (double) origin.element();
                IComplexNumber result = otherValue.mul(element);
                return NDArrays.scalar(result);

            }


        }

        else {
            //complex + real
            if(origin instanceof IComplexNDArray) {
                IComplexNDArray originComplexValue = (IComplexNDArray) origin;
                IComplexNumber firstValue = (IComplexNumber) originComplexValue.element();
                double realValue = (double) value.element();
                IComplexNumber retValue = firstValue.mul(realValue);
                return NDArrays.scalar(retValue);

            }

            //both normal
            else {
                double firstElement = (double) origin.element();
                double secondElement = (double) value.element();
                return NDArrays.scalar(firstElement * secondElement,0);
            }


        }
    }


}
