package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseElementWiseOp;

/**
 *
 * Max out activation:
 * http://arxiv.org/pdf/1302.4389.pdf
 *
 *
 * @author Adam Gibson
 *
 */
public class MaxOut extends BaseElementWiseOp {

    /**
     * Apply the transformation
     */
    @Override
    public void exec() {
        INDArray linear = from.linearView();
        if(linear instanceof IComplexNDArray) {
            IComplexNDArray cLinear = (IComplexNDArray) linear;
            int max = Nd4j.getBlasWrapper().iamax(cLinear);
            IComplexNumber max2 = cLinear.getComplex(max);
            for(int i = 0; i < cLinear.length(); i++) {
                cLinear.putScalar(i,max2);
            }

        }

        else {
            int max = Nd4j.getBlasWrapper().iamax(linear);
            float maxNum = linear.get(max);
            for(int i = 0; i < linear.length(); i++) {
                from.putScalar(i,maxNum);
            }
        }
    }

    /**
     * The transformation for a given value (a scalar)
     *
     * @param origin the origin ndarray
     * @param value  the value to apply (a scalar)
     * @param i      the index of the element being acted upon
     * @return the transformed value based on the input
     */
    @Override
    public <E> E apply(INDArray origin, Object value, int i) {
        return null;
    }
}
