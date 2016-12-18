package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationLReLU implements IActivation {

    private double alpha;

    public ActivationLReLU() {
        this.alpha = 0.01;
    }

    public ActivationLReLU (double alpha) {
        this.alpha = alpha;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,alpha));
        return in;
    }

    @Override
    public INDArray getGradient(INDArray in) {
        Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,alpha).derivative());
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
        return "leakyrelu";
    }
}
