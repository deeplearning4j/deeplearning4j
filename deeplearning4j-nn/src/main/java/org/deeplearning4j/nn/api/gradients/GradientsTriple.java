package org.deeplearning4j.nn.api.gradients;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

public class GradientsTriple extends BaseGradients {

    private INDArray actGrad1;
    private INDArray actGrad2;
    private INDArray actGrad3;

    public GradientsTriple(INDArray actGrad1, INDArray actGrad2, INDArray actGrad3, Gradient gradient){
        super(gradient);
        this.actGrad1 = actGrad1;
        this.actGrad2 = actGrad2;
        this.actGrad3 = actGrad3;
    }

    @Override
    public int size() {
        return 3;
    }

    @Override
    public INDArray getActivationGrad(int idx) {
        assertIndex(idx);
        switch (idx){
            case 0:
                return actGrad1;
            case 1:
                return actGrad2;
            case 2:
                return actGrad3;
            default:
                throw new RuntimeException();
        }
    }

    @Override
    public void setActivationGrad(int idx, INDArray actGrad) {
        assertIndex(idx);
        switch (idx){
            case 0:
                actGrad1 = actGrad;
                return;
            case 1:
                actGrad2 = actGrad;
                return;
            case 2:
                actGrad3 = actGrad;
        }
    }

}
