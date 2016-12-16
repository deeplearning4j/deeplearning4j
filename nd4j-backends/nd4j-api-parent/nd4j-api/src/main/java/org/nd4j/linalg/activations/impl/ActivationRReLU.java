package org.nd4j.linalg.activations.impl;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.PReLU;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.List;

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
        //FIXME: should go into libnd4j with leaky relu as a broadcast operation, push next release
        //Nd4j.getExecutioner().execAndReturn(new PReLU(in, alpha, activation));
        activation.assign(in);
        BooleanIndexing.replaceWhere(activation, 0.0, Conditions.greaterThan(0.0));
        BooleanIndexing.replaceWhere(activation, 1.0, Conditions.lessThan(0.0));
        activation.muli(this.alpha);
        activation.addi(Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in)));
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
        //assert alpha is the same shape as in
        //FIXME: derivative for prelu derivative in libnd4j
        //Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha,gradient).derivative());
        gradient.assign(in);
        BooleanIndexing.replaceWhere(gradient, 1.0, Conditions.greaterThanOrEqual(0.0));
        BooleanIndexing.replaceWhere(gradient, this.alpha, Conditions.lessThan(0));
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
        setActivation(in,activation);
        setGradient(in,gradient);
    }

}
