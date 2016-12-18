package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

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
    public INDArray getActivation(INDArray in, boolean training) {
        if (training) {
            this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
        }
        else {
            this.alpha = Nd4j.valueArrayOf(in.shape(),0.5*(l+u));
        }
        getActivation(in);
        return in;
    }

    private void getActivation(INDArray in){
        //FIXME: should go into libnd4j with leaky relu as a broadcast operation, push next release
        //Nd4j.getExecutioner().execAndReturn(new PReLU(in, alpha, activation));
        BooleanIndexing.replaceWhere(in, this.alpha, Conditions.lessThan(0));
    }

    @Override
    public INDArray getGradient(INDArray in) {
        //assert alpha is the same shape as in
        //FIXME: derivative for prelu derivative in libnd4j
        //Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha,gradient).derivative());
        BooleanIndexing.replaceWhere(in, 1.0, Conditions.greaterThanOrEqual(0.0));
        BooleanIndexing.replaceWhere(in, this.alpha, Conditions.lessThan(0));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> getActivationAndGradient(INDArray in) {
        INDArray activation = in.dup();
        INDArray gradient = in.dup();
        getActivation(activation, true);
        getGradient(gradient);
        return new Pair<INDArray, INDArray>(activation,gradient);
    }

    @Override
    public String toString() {
        return "rrelu";
    }

}
