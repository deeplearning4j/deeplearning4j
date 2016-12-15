package org.nd4j.linalg.activations.impl;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationIdentity implements IActivation {

    @Override
    public void setActivation(INDArray in, INDArray activation, boolean training) {
        activation = in.dup();
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
        gradient.muli(0).addi(1);
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        setActivation(in,activation, true);
        setGradient(in,gradient);
    }

    //Boilerplate code - not valid for functions that don't have learnable parameters
    @Override
    public int getNumParams() {
        return 0;
    }

    @Override
    public boolean[] isSharedParam() {
        return null;
    }

    @Override
    public boolean[] isShardedParam() {
        return null;
    }

    @Override
    public double[] getDefaultParamVals() {
        return null;
    }

    @Override
    public INDArray initParam(int paramIndex, int[] ofShape) {
        return null;
    }

    @Override
    public void setParams(double[] paramsShared, List<INDArray> paramsSharded) {

    }

    @Override
    public void setGradientParam(INDArray in, int paramIndex, INDArray gradient) {

    }

    @Override
    public int[] getShardAcrossDim() {
        return null;
    }
}
