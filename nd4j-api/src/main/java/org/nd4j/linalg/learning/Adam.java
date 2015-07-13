package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;

/**
 * The Adam updater.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Adam Gibson
 */
public class Adam implements Serializable {

    private double alpha = 1e-3;
    private double beta1 = 0.9;
    private double beta2 = 0.99;
    private double lam = 1 - 1e-8;
    private double lr = -1;
    private double beta1T = -1;
    private INDArray m,v;

    public Adam(double alpha, double beta1, double beta2, double lam, double lr, double beta1T) {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.lam = lam;
        this.lr = lr;
        this.beta1T = beta1T;
    }

    /**
     * Calculate the update based
     * on the given gradient
     * @param gradient the gradient to get the update for
     * @return the gradient
     */
    public INDArray getGradient(INDArray gradient) {
        if(m == null) {
            m = Nd4j.zeros(gradient.shape());
        }

        if(beta1T < 0)
            beta1T = beta1T();


        m.addi(1 - beta1T).muli(gradient.sub(m));
        if (v == null)
            v = Nd4j.zeros(gradient.shape());
        v.addi(1 - beta2).muli(gradient.mul(gradient).subi(v));
        if(lr < 0)
            lr = lr();

        return m.mul(lr).divi(Transforms.sqrt(v).add(Nd4j.EPS_THRESHOLD));
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

    public double getLam() {
        return lam;
    }

    public void setLam(double lam) {
        this.lam = lam;
    }

    public double getLr() {
        return lr;
    }

    public void setLr(double lr) {
        this.lr = lr;
    }

    public double getBeta1T() {
        return beta1T;
    }

    public void setBeta1T(double beta1T) {
        this.beta1T = beta1T;
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

    private double beta1T() {
        double t = 0.0;
        return beta1 * (Math.pow(lam ,(t - 1)));
    }


    private double lr() {
        double t = 0.0;
        double fix1 = 1. - Math.pow(beta1,t);
        double fix2 = 1 - Math.pow(beta2,t);
        return alpha * Math.sqrt(fix2) / fix1;
    }

}
