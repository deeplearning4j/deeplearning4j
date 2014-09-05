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
        INDArray rSub = input.rsub(1);
        return input.mul(rSub);
    }



}
