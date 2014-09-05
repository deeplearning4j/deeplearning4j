package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.ElementWiseOp;


/**
 * The exponential activation function
 * @author Adam Gibson
 */
public class Exp extends BaseActivationFunction {

    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public Class<? extends ElementWiseOp> transformClazz() {
        return  org.nd4j.linalg.ops.transforms.Exp.class;
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
        return apply(input);
    }


}
