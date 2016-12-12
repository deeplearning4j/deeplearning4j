package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLUDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationLeakyReLU implements IActivation{

    private double alpha;

    public ActivationLeakyReLU() {
       this.alpha = 0.1;
    }

    public ActivationLeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public INDArray computeActivation(INDArray in, boolean training) {
        computeActivation(in);
    }

    private INDArray computeActivation(INDArray in){
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,alpha));
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLUDerivative(in,alpha));
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
        return "leakyrelu";
    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public void setParamsViewArray(INDArray paramView) {

    }

    @Override
    public void setBackpropViewArray(INDArray in, INDArray params) {

    }
}
