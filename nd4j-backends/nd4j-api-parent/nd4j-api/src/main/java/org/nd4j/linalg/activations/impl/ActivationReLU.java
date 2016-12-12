package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationReLU implements IActivation{

    @Override
    public INDArray computeActivation(INDArray in, boolean training) {
        return computeActivation(in);
    }

    private INDArray computeActivation(INDArray in){
        return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in));
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in).derivative());
    }

    @Override
    public Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in) {
        return new Pair<INDArray, INDArray>(
                computeActivation(in,true),
                computeGradient(in)
        );
    }

    @Override
    public String toString() {
        return "relu";
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
