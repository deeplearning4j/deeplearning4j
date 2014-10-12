package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.BaseElementWiseOp;
import org.nd4j.linalg.util.BigDecimalMath;
import org.nd4j.linalg.util.ComplexUtil;

import java.math.BigDecimal;

/**
 * Exponential of an ndarray
 * @author Adam Gibson
 */
public class Exp extends BaseElementWiseOp {
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
            IComplexNumber c = (IComplexNumber) value;
            return  ComplexUtil.exp(c);
        }
        else {
            if(from.data().dataType().equals(DataBuffer.FLOAT)) {
                float val = (float) value;
                return (float) Math.exp(val);
            }
            else {
                double val = (double) value;
                if (val < 0) {
                    BigDecimal bigDecimal = BigDecimalMath.exp(BigDecimal.valueOf(val));
                    double val2 = bigDecimal.doubleValue();
                    return val2;
                }

                else
                    return Math.exp(val);


            }

        }

    }
}
