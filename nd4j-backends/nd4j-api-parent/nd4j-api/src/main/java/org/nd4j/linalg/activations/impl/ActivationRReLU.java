package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.PReLU;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationRReLU implements IActivation {

    private double l,u;
    INDArray alpha; //don't need to write to json, when streaming
    //private long seed; //repeatability??

    public ActivationRReLU() {
        this.l = 1/8;
        this.u = 1/3;
    }

    public ActivationRReLU(double l, double u) {
        this.l = l;
        this.u = u;
    }

    @Override
    public void setActivation(INDArray in, INDArray activation, boolean training) {
        if (training) {
            this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
        }
        else {
            this.alpha = Nd4j.valueArrayOf(in.shape(),0.5*(l+u));
        }
        setActivation(in,activation);
    }

    private void setActivation(INDArray in,INDArray activation){
        Nd4j.getExecutioner().execAndReturn(new PReLU(in, alpha, activation));
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
        //assert alpha is the same shape as in
        Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha,gradient).derivative());
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
        setActivation(in,activation);
        setGradient(in,gradient);
    }
}
