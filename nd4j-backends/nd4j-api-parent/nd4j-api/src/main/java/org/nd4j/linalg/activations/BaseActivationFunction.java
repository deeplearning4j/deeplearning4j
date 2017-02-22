package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Base IActivation for activation functions without parameters
 *
 * @author Alex Black
 */
public abstract class BaseActivationFunction implements IActivation {

    @Override
    public int numParams(int inputSize) {
        return 0;
    }

    @Override
    public void setParametersViewArray(INDArray viewArray, boolean initialize) {
        //No op
    }

    @Override
    public INDArray getParametersViewArray() {
        return null;
    }

    @Override
    public void setGradientViewArray(INDArray viewArray) {
        //No op
    }

    @Override
    public INDArray getGradientViewArray() {
        return null;
    }
}
