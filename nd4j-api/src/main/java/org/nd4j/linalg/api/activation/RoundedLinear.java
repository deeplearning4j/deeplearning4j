package org.nd4j.linalg.api.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

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
    public ElementWiseOpFactory transformFactory() {
        return ElementWiseOpFactories.round();
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
