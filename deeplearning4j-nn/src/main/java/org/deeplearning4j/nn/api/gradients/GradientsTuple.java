package org.deeplearning4j.nn.api.gradients;

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;


public class GradientsTuple extends BaseGradients {

    private INDArray[] actGrads;

    public GradientsTuple(INDArray[] actGrads, Gradient paramGrad){
        super(paramGrad);
        this.actGrads = actGrads;
    }

    @Override
    public int size() {
        return actGrads.length;
    }

    @Override
    public INDArray getActivationGrad(int idx) {
        assertIndex(idx);
        return actGrads[idx];
    }

    @Override
    public void setActivationGrad(int idx, INDArray activationGradient) {
        assertIndex(idx);
        actGrads[idx] = activationGradient;
    }
}
