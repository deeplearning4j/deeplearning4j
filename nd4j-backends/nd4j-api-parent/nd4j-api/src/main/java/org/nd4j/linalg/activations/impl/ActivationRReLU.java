package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.PReLU;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationRReLU implements IActivation{

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
    public INDArray computeActivation(INDArray in, boolean training) {
        if (training) {
            this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
        }
        else {
            this.alpha = Nd4j.valueArrayOf(in.shape(),0.5*(l+u));
        }
        return computeActivation(in);
    }

    private INDArray computeActivation(INDArray in){
        return Nd4j.getExecutioner().execAndReturn(new PReLU(in, alpha));
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        //assert alpha is the same shape as in
        return Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha).derivative());
    }

    @Override
    public Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in) {
        //assumption is this is training, since the gradient is also calculated
        this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
        return new Pair<INDArray, INDArray>(
                computeActivation(in),
                computeGradient(in)
        );
    }

    @Override
    public String toString() {
        return "rrelu";
    }

}
