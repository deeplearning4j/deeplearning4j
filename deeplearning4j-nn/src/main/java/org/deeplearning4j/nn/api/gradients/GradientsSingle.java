package org.deeplearning4j.nn.api.gradients;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

public class GradientsSingle extends BaseGradients {

    private INDArray actGrad;   //AKA "epsilon", dL/da

    public GradientsSingle(INDArray actGrad, Gradient gradient){
        super(gradient);
        this.actGrad = actGrad;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public INDArray getActivationGrad(int idx) {
        assertIndex(idx);
        return actGrad;
    }

    @Override
    public void setActivationGrad(int idx, INDArray activationGradient) {
        assertIndex(idx);
        actGrad = activationGradient;
    }
}
