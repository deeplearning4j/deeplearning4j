package org.deeplearning4j.linalg.api.activation;


import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.ElementWiseOp;

/**
 * Sigmoid function (complex AND real!)
 *
 * http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1742-18.pdf
 *
 * For complex we set the k = 1
 *
 *
 * @author Adam Gibson
 */
public class Sigmoid extends BaseActivationFunction {


    /**
     *
     */
    private static final long serialVersionUID = -6280602270833101092L;
    private int k = 1;

    @Override
    public INDArray apply(INDArray input) {
        if(input instanceof IComplexNDArray) {
            IComplexNumber number = (IComplexNumber) input.element();
            double arg = number.arg();
            double sigArg = 1  / 1 + (Math.exp(-arg)) - k + .5;
            double ret = Math.exp(sigArg);
            return NDArrays.scalar(NDArrays.createDouble(ret,0));

        }
        else
            return  NDArrays.scalar(1 / 1 + Math.exp(-(double) input.element()));
    }

    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public Class<? extends ElementWiseOp> transformClazz() {
        return org.deeplearning4j.linalg.ops.transforms.Sigmoid.class;
    }

    @Override
    public INDArray applyDerivative(INDArray input) {
        return input.mul(input.rsubi(NDArrays.ones(input.shape())));
    }



}
