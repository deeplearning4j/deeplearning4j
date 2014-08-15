package org.deeplearning4j.linalg.api.activation;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.ElementWiseOp;

/**
 * Rounded output
 * @author Adam Gibson
 */
public class RoundedLinear extends BaseActivationFunction {


    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public Class<? extends ElementWiseOp> transformClazz() {
        return org.deeplearning4j.linalg.ops.transforms.Round.class;
    }

    /**
     * Applies the derivative of this function
     *
     * @param input the input to apply it to
     * @return the derivative of this function applied to
     * the input
     */
    @Override
    public INDArray applyDerivative(INDArray input) {
        return NDArrays.ones(input.shape());
    }



}
