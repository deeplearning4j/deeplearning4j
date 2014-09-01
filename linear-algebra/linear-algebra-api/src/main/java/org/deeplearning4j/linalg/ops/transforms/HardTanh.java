package org.deeplearning4j.linalg.ops.transforms;

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
    public Object apply(INDArray from,Object value, int i) {
        if(value instanceof IComplexNumber) {
            IComplexNumber element = (IComplexNumber) value;
            IComplexNumber ret = ComplexUtil.tanh(element);
            if(ret.realComponent().floatValue() < -1)
                ret.set(-1,ret.imaginaryComponent().floatValue());
            if(ret.realComponent().floatValue() > 1)
                ret.set(1,ret.imaginaryComponent().floatValue());
            return NDArrays.scalar(ret);
        }
        else  {
            float d = (float) value;
            float ret = (float) Math.tanh(d);
            if(ret < -1)
                return -1;
            else if(ret > 1)
                return 1;
            else
                return ret;
        }
    }
}