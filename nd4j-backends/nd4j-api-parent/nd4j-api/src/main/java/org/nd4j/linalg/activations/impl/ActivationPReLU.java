package org.nd4j.linalg.activations.impl;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.PReLU;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationPReLU implements IActivation {

    private static final int NUM_PARAMS = 1;
    private static final boolean [] SHARE_ALL = {false};
    private static final boolean [] SHARD_ALL = {true};
    private INDArray alpha;
    private double [] defaultVal = {0.01};

    public ActivationPReLU (double alpha) {
        this.defaultVal = new double[] {alpha};
    }

    @Override
    public void setParams(double [] paramsShared, List<INDArray> paramsSharded) {
        this.alpha = paramsSharded.get(0);
    }

    @Override
    public void setActivation(INDArray in, INDArray activation, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha,activation));
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
        Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha,gradient).derivative());
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        setActivation(in,activation,true);
        setGradient(in,gradient);
    }

    @Override
    public void setGradientParam(INDArray in, int paramIndex, INDArray gradient) {
        Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha,gradient).derivative());
        gradient.divi(alpha).muli(in);
    }

    @Override
    public int getNumParams() {
        return 1;
    }

    @Override
    public boolean [] isSharedParam() {
        return SHARE_ALL;
    }

    @Override
    public boolean [] isShardedParam() {
        return SHARD_ALL;
    }

    @Override
    public double [] getDefaultParamVals() {
        return defaultVal;
    }

    @Override
    public INDArray initParam(int paramIndex, int[] ofShape) {
        return Nd4j.valueArrayOf(ofShape,defaultVal[0]);
    }

    @Override
    public int [] getShardAcrossDim() {
        return null;
    }

}
