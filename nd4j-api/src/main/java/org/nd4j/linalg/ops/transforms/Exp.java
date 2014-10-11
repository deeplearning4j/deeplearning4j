package org.nd4j.linalg.ops.transforms;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.BaseElementWiseOp;
import org.nd4j.linalg.util.BigDecimalMath;
import org.nd4j.linalg.util.ComplexUtil;

import java.math.BigDecimal;

/**
 * Exponential of an ndarray
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
            float val = (float) value;
            if(val < 0) {
                BigDecimal bigDecimal = BigDecimalMath.exp(BigDecimal.valueOf(val));
                float retVal = bigDecimal.floatValue();
                return retVal;
            }

            double exp = Math.expm1(val);
            return (float) Math.expm1(val);

        }

    }
}
