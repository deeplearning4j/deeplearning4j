package org.deeplearning4j.nn.api.gradients;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

public class GradientsPair extends BaseGradients {

    private INDArray actGrad1;
    private INDArray actGrad2;

    public GradientsPair(INDArray actGrad1, INDArray actGrad2, Gradient gradient){
        super(gradient);
        this.actGrad1 = actGrad1;
        this.actGrad2 = actGrad2;
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public INDArray getActivationGrad(int idx) {
        assertIndex(idx);
        return (idx == 0 ? actGrad1 : actGrad2);
    }

    @Override
    public void setActivationGrad(int idx, INDArray actGrad) {
        assertIndex(idx);
        if(idx == 0){
            actGrad1 = actGrad;
        } else {
            actGrad2 = actGrad;
        }
    }

}
