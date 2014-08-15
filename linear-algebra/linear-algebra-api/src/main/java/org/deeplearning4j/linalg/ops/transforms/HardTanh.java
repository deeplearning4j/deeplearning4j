package org.deeplearning4j.linalg.ops.transforms;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseElementWiseOp;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 * Hard Tanh
 *
 * values range: -1 < tanh(x) < 1
 *
 * @author Adam Gibson
 */
public class HardTanh extends BaseElementWiseOp {


    /**
     * The transformation for a given value (a scalar ndarray)
     *
     * @param value the value to applyTransformToOrigin (a scalar ndarray)
     * @param i     the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public INDArray apply(INDArray value, int i) {
        if(value instanceof IComplexNDArray) {
            IComplexNumber element = (IComplexNumber) value.element();
            IComplexNumber ret = ComplexUtil.tanh(element);
            if(ret.realComponent().doubleValue() < -1)
                ret.set(-1,ret.imaginaryComponent().doubleValue());
            if(ret.realComponent().doubleValue() > 1)
                ret.set(1,ret.imaginaryComponent().doubleValue());
            return NDArrays.scalar(ret);
        }
        else  {
            double d = (double) value.element();
            double ret = Math.tanh(d);
            if(ret < -1)
                return NDArrays.scalar(-1);
            else if(ret > 1)
                return NDArrays.scalar(1);
            else
                return NDArrays.scalar(ret);
        }
    }
}