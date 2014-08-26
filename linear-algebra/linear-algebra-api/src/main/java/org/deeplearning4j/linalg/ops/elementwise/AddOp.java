package org.deeplearning4j.linalg.ops.elementwise;


import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class AddOp extends BaseTwoArrayElementWiseOp {



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
            IComplexNumber firstValue = (IComplexNumber) complexValue.element();
            //complex + complex
            if(origin instanceof IComplexNDArray) {
                IComplexNDArray originComplex = (IComplexNDArray) origin;
                IComplexNumber originValue = (IComplexNumber) originComplex.element();
                return NDArrays.scalar(originValue.add(firstValue));
            }

            //real + complex
            else {
                float element = (float) origin.element();
                IComplexNumber result = firstValue.add(element);
                return NDArrays.scalar(result);

            }


        }

        else {
            //complex + real
            if(origin instanceof IComplexNDArray) {
                IComplexNDArray complexValue = (IComplexNDArray) origin;
                IComplexNumber firstValue = (IComplexNumber) complexValue.element();
                float realValue = (float) value.element();
                IComplexNumber retValue = firstValue.add(realValue);
                return NDArrays.scalar(retValue);

            }

            //both normal
            else {
                float firstElement = (float) value.element();
                float secondElement = (float) origin.element();
                return NDArrays.scalar(firstElement + secondElement,0);
            }


        }
    }


}
