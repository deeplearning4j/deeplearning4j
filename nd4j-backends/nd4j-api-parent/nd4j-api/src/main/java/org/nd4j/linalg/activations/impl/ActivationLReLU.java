package org.nd4j.linalg.activations.impl;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

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
    public void setActivation(INDArray in, INDArray activation, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,activation,alpha));
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
        Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,gradient,alpha).derivative());
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        setActivation(in,activation,true);
        setGradient(in,gradient);
    }
}
