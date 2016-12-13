package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationIdentity implements IActivation {

    @Override
    public INDArray computeActivation(INDArray in, boolean training) {
        return computeActivation(in);
    }

    private INDArray computeActivation(INDArray in){
        return in.dup();
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        return Nd4j.ones(in.shape());
    }

    @Override
    public Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in) {
        return new Pair<INDArray, INDArray>(
                computeActivation(in),
                computeGradient(in)
        );
    }

    @Override
    public String toString() {
        return "identity";
    }

}
