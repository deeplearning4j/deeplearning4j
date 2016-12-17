package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationIdentity implements IActivation {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        //no op
        return in;
    }

    @Override
    public INDArray getGradient(INDArray in) {
        in.muli(0).addi(1);
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

}
