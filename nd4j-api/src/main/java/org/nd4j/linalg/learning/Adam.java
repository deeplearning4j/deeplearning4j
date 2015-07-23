package org.nd4j.linalg.learning;

import org.apache.commons.math3.util.FastMath;
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
public class Adam implements Serializable,GradientUpdater {

    private double alpha = 1e-3;
    private double beta1 = 0.9;
    private double beta2 = 0.99;
    private double lam = 1 - 1e-8;
    private double lr = -1;
    private double beta1T = -1;
    private INDArray m,v;
    private double t = 1;

    public Adam() {
    }

    /**
     *
     * @param alpha
     * @param beta1
     * @param beta2
     * @param lam
     * @param lr
     * @param beta1T
     */
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
     * @param iteration
     * @return the gradient
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(m == null) {
            m = Nd4j.zeros(gradient.shape());
        }



        this.t = iteration;
        beta1T = beta1T();


        m.addi(1 - beta1T).muli(gradient.sub(m));
        if (v == null)
            v = Nd4j.zeros(gradient.shape());
        v.addi(1 - beta2).muli(gradient.mul(gradient).subi(v));
        lr = lr();
        INDArray mTimesLr = m.mul(lr);
        INDArray sqrtV = Transforms.sqrt(v).add(Nd4j.EPS_THRESHOLD);
        Nd4j.clearNans(sqrtV);
        return mTimesLr.divi(sqrtV);
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
        return beta1 * (Math.pow(lam ,(t - 1)));
    }


    private double lr() {
        double fix1 = 1. - Math.pow(beta1,t);
        double fix2 = 1 - Math.pow(beta2,t);
        double ret =  alpha * FastMath.sqrt(fix2) / fix1;
        if(Double.isNaN(ret))
            return Nd4j.EPS_THRESHOLD;
        return ret;
    }

}
