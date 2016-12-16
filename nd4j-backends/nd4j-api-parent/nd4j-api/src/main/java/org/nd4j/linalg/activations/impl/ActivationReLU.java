package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

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
