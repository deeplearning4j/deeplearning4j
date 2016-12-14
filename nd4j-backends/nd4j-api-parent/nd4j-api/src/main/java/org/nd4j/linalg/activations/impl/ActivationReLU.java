package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/13/16.
 */
public class ActivationReLU implements IActivation {

    @Override
    public void setActivation(INDArray in, INDArray activation, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in,activation));
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
        Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in,gradient).derivative());
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        setActivation(in,activation, true);
        setGradient(in,gradient);
    }

}
