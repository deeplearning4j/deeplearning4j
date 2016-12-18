package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationIdentity extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        //no op
        return in;
    }

    @Override
    public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon) {
        return new Pair<>(epsilon, null);
    }

    @Override
    public String toString() {
        return "identity";
    }

}
