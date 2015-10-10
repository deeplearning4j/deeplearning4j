package org.nd4j.linalg.learning;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;

import lombok.NoArgsConstructor;

/**
 * The Adam updater.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class Adam implements Serializable,GradientUpdater {

    private double alpha = 1e-3; // learning rate
    private double beta1 = 0.9; // gradient moving avg decay rate
    private double beta2 = 0.999; // gradient sqrd decay rate
    private double epsilon = 1e-8;
    private INDArray m,v; // moving avg & sqrd gradients

    public Adam(double alpha, double beta1, double beta2, double epsilon) {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon; // fudge factor to avoid zeros
    }

    public Adam(double alpha, double beta1, double beta2) {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    public Adam(double alpha) {
        this.alpha = alpha;
    }

    /**Calculate the update based on the given gradient
     * @param gradient the gradient to get the update for
     * @param iteration
     * @return the gradient
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(m == null) m = Nd4j.zeros(gradient.shape());
        if (v == null) v = Nd4j.zeros(gradient.shape());

        INDArray oneMinusBeta1Grad = gradient.mul(1.0-beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1-beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration);
        double beta2t = FastMath.pow(beta2, iteration);

        double alphat = alpha * FastMath.sqrt(1-beta2t)/(1-beta1t);
        if(Double.isNaN(alphat) || alphat==0.0) alphat = Nd4j.EPS_THRESHOLD;
        INDArray sqrtV = Transforms.sqrt(v).addi(epsilon);
        INDArray ret = m.mul(alphat).divi(sqrtV);
        return ret;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getBeta1() {
        return beta1;
    }

    public void setBeta1(double beta1) {
        this.beta1 = beta1;
    }

    public double getBeta2() {
        return beta2;
    }

    public void setBeta2(double beta2) {
        this.beta2 = beta2;
    }

    public INDArray getM() {
        return m;
    }

    public void setM(INDArray m) {
        this.m = m;
    }

    public INDArray getV() {
        return v;
    }

    public void setV(INDArray v) {
        this.v = v;
    }
}
