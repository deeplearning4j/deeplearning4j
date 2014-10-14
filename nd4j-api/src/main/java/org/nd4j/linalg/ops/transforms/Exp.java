package org.nd4j.linalg.ops.transforms;

import org.apache.commons.math3.util.FastMath;
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
                double val = (double) value;
                return FastMath.exp(val);
            }
            else {
                double val = (double) value;
                if (val < 0) {
                    double ret = FastMath.exp(val);
                    return  ret;
                }

                else
                    return Math.exp(val);


            }

        }

    }


    public  double exp(double val) {
        final long tmp = (long) (1512775 * val) + 1072693248;
        final long mantissa = tmp & 0x000FFFFF;
        int error = (int) mantissa >> 7;   // remove chance of overflow
        error = (int) (error - mantissa * mantissa) / 186; // subtract mantissa^2 * 64
        // 64 / 186 = 1/2.90625
        return Double.longBitsToDouble((tmp - error) << 32);
    }
}
