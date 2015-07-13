package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class SgdUpdater implements GradientUpdater {
    private double lr = 1e-1;

    public SgdUpdater(double lr) {
        this.lr = lr;
    }

    public double getLr() {
        return lr;
    }

    public void setLr(double lr) {
        this.lr = lr;
    }

    @Override
    public INDArray getGradient(INDArray gradient) {
        return gradient.mul(lr);
    }
}
