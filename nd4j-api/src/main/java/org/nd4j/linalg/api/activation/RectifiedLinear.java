package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ElementWiseOp;

/**
 * Rectified Linear
 * max(0,input)
 * @author Adam Gibson
 */
public class RectifiedLinear extends BaseActivationFunction {


    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public Class<? extends ElementWiseOp> transformClazz() {
        return org.nd4j.linalg.ops.transforms.Max.class;
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
        return Nd4j.ones(input.shape());
    }



}
